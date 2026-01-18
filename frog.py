import torch
from typing import Iterable, Callable
from torch.nn import grad as nn_grad


@torch.no_grad()
def batched_cg_conv(X_act, Y2_scaled, G, module: torch.nn.Conv2d, num_iters: int, eps=1e-6, denom_eps=1e-10):
    """
    Solve (X_act^T @ Y2_scaled @ X_act + eps * I)W = G for every output channel (row-wise) using CG.
    """

    W = torch.zeros_like(G)
    R = G.clone()
    P = R.clone()

    rsold = (R * R).sum(dim=(1, 2, 3), keepdim=True)

    for _ in range(num_iters):
        XP = torch.nn.functional.conv2d(
            X_act, P,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        XP.mul_(Y2_scaled)
        Ap = nn_grad.conv2d_weight(
            X_act, P.shape, XP,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        Ap.add_(P, alpha=eps)

        pAp = (P * Ap).sum(dim=(1, 2, 3), keepdim=True)
        alpha = rsold / pAp.add_(denom_eps)

        W.addcmul_(P, alpha)
        R.addcmul_(Ap, alpha, value=-1.0)

        rsnew = (R * R).sum(dim=(1, 2, 3), keepdim=True)
        beta = rsnew / rsold.add_(denom_eps)

        P.mul_(beta).add_(R)
        rsold = rsnew

    return W


@torch.no_grad()
def fisher_normalized_trace(X, Y2dB, module, ones):
    """
    Compute normalized trace of row-wise Fisher matrices.
    """
    assert module.groups == 1, "FROG currently only supports groups=1"

    c_out, c_in, kh, kw = module.weight.shape
    x2_sum = torch.linalg.vecdot(X, X, dim=1)
    window_sum = torch.nn.functional.conv2d(
        x2_sum.unsqueeze(1),
        ones,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation
    )
    scalar = (Y2dB * window_sum).sum(dim=(0, 2, 3))
    normalization = c_in * kh * kw
    return scalar.div_(normalization)


class Frog(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[torch.Tensor],
            model: torch.nn.Module,
            lr: float = 1e-3,
            lr_other_ratio: float = 0.25,
            momentum: float = 0.0,
            nesterov: bool = False,
            weight_decay: float = 0.0,
            sample_size: int | None = 24,
            num_iters: int = 4,
            tau: float = 1e-6,
            batch_averaged: bool = True,
            skip_modules: list[str] = None,
            denom_eps: float = 1e-10
    ):
        """
        FROG: Fisher ROw-wise PreconditioninG.

        Current implementation:
          - Conv2d only (Linear not supported yet)
          - Iterative solve for row-wise empirical Fisher matrices using batched CG.
          - Random subsampling of activations (X and Y)

        :param params: Optimized parameters
        :param model: PyTorch model (required for hooks)
        :param lr: Learning rate for preconditioned modules
        :param lr_other_ratio: Ratio for learning rate of other modules (lr_other = lr * lr_other_ratio)
        :param momentum: Momentum (same as in SGD)
        :param nesterov: Nesterov momentum (same as in SGD)
        :param weight_decay: Weight decay (same as in SGD)
        :param sample_size: Number of activations to subsample for iterative CG. None means keep all activations (not recommended).
        :param num_iters: Number of CG iterations
        :param tau: Fisher damping factor: damping = tr(F)/D * tau
        :param batch_averaged: Whether loss has reduction="mean" (for rescaling)
        :param skip_modules: Names of modules to skip
        :param denom_eps: denominator epsilon for numerical stability
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_other_ratio < 0.0:
            raise ValueError(f"Invalid lr_other_ratio: {lr_other_ratio}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0.0:
            raise ValueError("Nesterov requires momentum > 0.")
        if sample_size is not None and sample_size <= 0:
            raise ValueError(f"sample_size must be None or positive, got {sample_size}")
        if num_iters <= 0:
            raise ValueError(f"num_iters must be positive, got {num_iters}")
        if tau < 0:
            raise ValueError(f"Invalid tau value: {tau}")
        if denom_eps < 0:
            raise ValueError(f"Invalid denominator epsilon value: {denom_eps}")

        defaults = dict(
            lr=lr,
            lr_other_ratio=lr_other_ratio,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            sample_size=sample_size,
            num_iters=num_iters,
            tau=tau,
            batch_averaged=batch_averaged,
            denom_eps=denom_eps
        )
        super().__init__(params, defaults)

        self.sample_indices: dict[torch.nn.Module, torch.Tensor] = {}
        self.out_grads: dict[torch.nn.Module, torch.Tensor] = {}
        self.in_acts: dict[torch.nn.Module, torch.Tensor] = {}
        self.param_to_module: dict[torch.Tensor, torch.nn.Module] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._ones_buffer = {}

        skip_modules = skip_modules or []

        for name, module in model.named_modules():
            if name not in skip_modules:
                if isinstance(module, torch.nn.Conv2d):
                    h_fwd = module.register_forward_hook(self._save_input)
                    h_bwd = module.register_full_backward_hook(self._save_grad_outputs)
                    self.hooks.extend([h_fwd, h_bwd])
                    self.param_to_module[module.weight] = module

    def _get_ones(self, kh, kw, device, dtype):
        key = (kh, kw, device, dtype)
        if key not in self._ones_buffer:
            self._ones_buffer[key] = torch.ones((1, 1, kh, kw), device=device, dtype=dtype)
        return self._ones_buffer[key]

    def _save_input(self, module: torch.nn.Module, input, output):
        if not torch.is_grad_enabled():
            return
        if len(input) == 0:
            return

        with torch.no_grad():
            X_all = input[0]

            if isinstance(module, torch.nn.Conv2d):
                B_all = X_all.shape[0]
                sample_size = self.defaults['sample_size']
                if sample_size is not None and sample_size < B_all:
                    idx = torch.randperm(X_all.shape[0], device=X_all.device)[:sample_size]
                    self.sample_indices[module] = idx
                    self.in_acts[module] = X_all[idx].detach()
                else:
                    self.in_acts[module] = X_all.detach().clone()

    def _save_grad_outputs(self, module: torch.nn.Module, grad_input, grad_output):
        if not grad_output or grad_output[0] is None:
            return

        with torch.no_grad():
            Y_all = grad_output[0]

            if isinstance(module, torch.nn.Conv2d):
                if module in self.sample_indices:
                    idx = self.sample_indices.pop(module)
                    Y = Y_all[idx].detach()
                else:
                    Y = Y_all.detach().clone()

                B_all = Y_all.shape[0]
                B = Y.shape[0]

                if self.defaults['batch_averaged']:
                    scale = (B_all * B_all) / B
                    Y2dB = Y.square_().mul_(scale)
                else:
                    Y2dB = Y.square_().div_(B)

                self.out_grads[module] = Y2dB


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lr_other_ratio = group['lr_other_ratio']
            momentum = group["momentum"]
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            sample_size = group['sample_size']
            num_iters = group['num_iters']
            tau = group['tau']
            batch_averaged = group['batch_averaged']
            denom_eps = group['denom_eps']

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Frog does not support sparse gradients")

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0.0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                state["step"] += 1

                if momentum > 0.0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    eff_grad = grad.add(buf, alpha=momentum) if nesterov else buf
                else:
                    eff_grad = grad

                if p in self.param_to_module:
                    module = self.param_to_module[p]

                    if isinstance(module, torch.nn.Linear):
                        raise Exception("Linear not supported yet")
                    elif isinstance(module, torch.nn.Conv2d):
                        X = self.in_acts[module]
                        Y2dB = self.out_grads[module]

                        F_tr = fisher_normalized_trace(X, Y2dB, module, ones=self._get_ones(p.shape[-2], p.shape[-1], X.device, X.dtype))
                        Y2_scaled = Y2dB.div_(F_tr.add_(denom_eps).view(1, -1, 1, 1))

                        # Solve row-wise (F/F_tr + tau * I)W = G using CG
                        P1 = batched_cg_conv(
                            X, Y2_scaled, eff_grad,
                            module=module,
                            num_iters=num_iters,
                            eps=tau,
                            denom_eps=denom_eps
                        )

                        # Normalization
                        P1_dot = (P1 * eff_grad).sum(dim=(1, 2, 3))
                        scalar = P1_dot.add_(denom_eps).rsqrt_()

                        if weight_decay > 0.0:
                            p.mul_(1 - lr * weight_decay)
                        p.addcmul_(P1, scalar.view(-1, 1, 1, 1), value=-lr)
                    else:
                        raise ValueError(f"Invalid module: {module}")
                else:
                    direction = eff_grad
                    if weight_decay > 0.0:
                        direction.add_(p, alpha=weight_decay)
                    p.add_(direction, alpha=-lr * lr_other_ratio)

        # Clear references
        self.in_acts.clear()
        self.out_grads.clear()
        self.sample_indices.clear()
        return loss

    def close(self):
        """
        Remove all hooks at the end of training
        """

        for h in self.hooks:
            h.remove()
        self.hooks.clear()
