import torch
from typing import Iterable, Callable
from torch.nn import grad as nn_grad


@torch.no_grad()
def _batched_cg(
        matmul: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        num_iters: int,
        denom_eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve A X = rhs for multiple RHS columns using batched Conjugate Gradient,
    where A is given implicitly via matmul(V) = A @ V.

    rhs: (n, m)
    """
    X = torch.zeros_like(rhs)
    R = rhs.clone()
    P = R.clone()

    rsold = (R * R).sum(dim=0)

    for _ in range(num_iters):
        Ap = matmul(P)
        pAp = (P * Ap).sum(dim=0)
        alpha = rsold / (pAp + denom_eps)
        alpha2d = alpha.view(1, -1)

        X.addcmul_(P, alpha2d)
        R.addcmul_(Ap, -alpha2d)

        rsnew = (R * R).sum(dim=0)
        beta = rsnew / (rsold + denom_eps)
        P.mul_(beta.view(1, -1)).add_(R)
        rsold = rsnew

    residuals = torch.sqrt(rsold)
    return X, residuals


@torch.no_grad()
def batched_cg_conv(X, G, module: torch.nn.Conv2d, num_iters: int, eps=1e-6):
    """
    Solve ( (X^T X)/B + eps*I ) W = G implicitly in conv-weight space.
    """

    if module.groups != 1:
        raise NotImplementedError("batched_cg_conv currently supports groups=1 only.")

    C_out, C_in, kh, kw = G.shape
    B, _, H, W = X.shape

    def matmul(V: torch.Tensor) -> torch.Tensor:
        v_kernels = V.transpose(0, 1).reshape(C_out, C_in, kh, kw)
        XV = torch.nn.functional.conv2d(X, v_kernels, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
        FV = nn_grad.conv2d_weight(X, v_kernels.shape, XV, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
        AV = FV.reshape(C_out, -1).transpose(0, 1)
        AV.div_(B)
        AV.add_(V, alpha=eps)
        return AV

    G_flat = G.reshape(C_out, -1).transpose(0, 1)
    X_flat, residuals = _batched_cg(matmul, G_flat, num_iters=num_iters)
    X_solution = X_flat.transpose(0, 1).reshape(C_out, C_in, kh, kw)
    return X_solution, residuals


@torch.no_grad()
def act_diag_conv(
        X: torch.Tensor,
        module: torch.nn.Conv2d
) -> torch.Tensor:
    """
    Compute diagonal (per-input-channel, per-kernel-entry) activation covariance:
      diag( (X^T X)/B ) in conv patch space.
    """

    if module.groups != 1:
        raise NotImplementedError("act_diag_conv currently supports groups=1 only.")

    B, C, H, W = X.shape
    kh, kw = module.kernel_size
    sh, sw = module.stride
    ph, pw = module.padding
    dh, dw = module.dilation
    H_out = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    W_out = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Invalid conv output size: H_out={H_out}, W_out={W_out} for input {(H, W)}")

    grad_out = torch.ones(
        (B, 1, H_out, W_out),
        device=X.device,
        dtype=X.dtype,
    )

    F = nn_grad.conv2d_weight(
        X * X,
        weight_size=(1, C, kh, kw),
        grad_output=grad_out,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups
    ).squeeze(0)

    F.div_(B)
    return F


class Frog(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[torch.Tensor],
            model: torch.nn.Module,
            lr: float = 1e-3,
            momentum: float = 0.0,
            nesterov: bool = False,
            y_momentum: float = 0.0,
            sample_size: int | None = 256,
            num_iters: int = 4,
            batch_averaged: bool = True,
            skip_modules: list[str] = None,
            y_sample_size: int | None = None,
            eps: float = 1e-6
    ):
        """
        FROG: Fisher ROw-wise PreconditioninG.

        Current implementation:
          - Conv2d only (Linear not supported yet)
          - Row-wise output scaling using diag of YY^T
          - Iterative solve for input-side factor via batched CG in conv-weight space
          - Random subsampling of batch statistics (sample_size and y_sample_size)

        :param params: Optimized parameters
        :param model: PyTorch model (required for hooks)
        :param lr: Learning rate for both preconditioned and unconditioned modules
        :param momentum: Momentum (same as in SGD)
        :param nesterov: Nesterov momentum (same as in SGD)
        :param y_momentum: Momentum for grad-output scaling (same as in Adam)
        :param sample_size: Number of activations to subsample for iterative CG. None means keep all activations
        :param num_iters: Number of CG iterations
        :param batch_averaged: Whether loss has reduction="mean" (for rescaling)
        :param skip_modules: Names of modules to skip
        :param y_sample_size: Number of grad-outputs to average across for diagonal scaling. None means across all (recommended)
        :param eps: epsilon for both CG and grad_output sqrt
        """

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0.0:
            raise ValueError("Nesterov requires momentum > 0.")
        if y_momentum < 0.0:
            raise ValueError(f"Invalid y_momentum value: {y_momentum}")
        if sample_size is not None and sample_size <= 0:
            raise ValueError(f"sample_size must be None or positive, got {sample_size}")
        if num_iters <= 0:
            raise ValueError(f"num_iters must be positive, got {num_iters}")
        if y_sample_size is not None and y_sample_size <= 0:
            raise ValueError(f"y_sample_size must be None or positive, got {y_sample_size}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            y_momentum=y_momentum,
            eps=eps,
            num_iters=num_iters,
            sample_size=sample_size,
            y_sample_size=y_sample_size,
            batch_averaged=batch_averaged
        )
        super().__init__(params, defaults)

        self.out_grads: dict[torch.Tensor, torch.Tensor] = {}
        self.in_acts: dict[torch.Tensor, torch.Tensor] = {}
        self.param_to_module: dict[torch.Tensor, torch.nn.Module] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

        skip_modules = skip_modules or []

        for name, module in model.named_modules():
            if name not in skip_modules:
                if isinstance(module, torch.nn.Conv2d):
                    h_fwd = module.register_forward_hook(self._save_input)
                    h_bwd = module.register_full_backward_hook(self._save_grad_outputs)
                    self.hooks.extend([h_fwd, h_bwd])
                    self.param_to_module[module.weight] = module

    def _save_input(self, module: torch.nn.Module, input, output):
        if not torch.is_grad_enabled():
            return
        if len(input) == 0:
            return

        X = input[0]

        if isinstance(module, torch.nn.Conv2d):
            self.in_acts[module.weight] = X.detach()

    def _save_grad_outputs(self, module: torch.nn.Module, grad_input, grad_output):
        if not grad_output or grad_output[0] is None:
            return

        Y = grad_output[0]

        if isinstance(module, torch.nn.Conv2d):
            self.out_grads[module.weight] = Y.detach()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group['nesterov']
            y_momentum = group['y_momentum']
            eps = group['eps']
            num_iters = group['num_iters']
            sample_size = group['sample_size']
            y_sample_size = group['y_sample_size']
            batch_averaged = group['batch_averaged']

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AWPFull does not support sparse gradients")

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0.0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    if y_momentum > 0.0 and (p in self.in_acts and p in self.out_grads):
                        state['y_momentum_buffer'] = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)

                state["step"] += 1

                if momentum > 0.0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    eff_grad = grad.add(buf, alpha=momentum) if nesterov else buf
                else:
                    eff_grad = grad

                # Preconditioning
                if p in self.in_acts and p in self.out_grads:
                    X_all = self.in_acts[p]
                    Y_all = self.out_grads[p]

                    module = self.param_to_module[p]

                    # Subsampling
                    if sample_size is not None and sample_size < X_all.shape[0]:
                        idx = torch.multinomial(torch.ones(X_all.shape[0], device=X_all.device), sample_size, replacement=False)
                        X = X_all[idx]
                    else:
                        X = X_all

                    if y_sample_size is not None and y_sample_size < Y_all.shape[0]:
                        idx = torch.multinomial(torch.ones(Y_all.shape[0], device=Y_all.device), y_sample_size, replacement=False)
                        Y = Y_all[idx]
                    else:
                        Y = Y_all

                    B = X.shape[0]

                    if isinstance(module, torch.nn.Linear):
                        raise NotImplementedError('Linear layers not yet supported')
                    elif isinstance(module, torch.nn.Conv2d):
                        y_diag = (Y ** 2).sum(dim=(0, 2, 3))  # C_out
                        # Rescaling
                        if batch_averaged:
                            y_diag.mul_(B)
                        else:
                            y_diag.div_(B)

                        if y_momentum > 0.0:
                            y_buf = state['y_momentum_buffer']
                            y_buf.mul_(y_momentum).add_(y_diag, alpha=1-y_momentum)
                            bias_correction = 1 - y_momentum ** state['step']
                            y_eff = y_buf / bias_correction
                        else:
                            y_eff = y_diag

                        # Left preconditioning
                        P1 = eff_grad / (y_eff + eps).sqrt_().view(-1, 1, 1, 1)

                        # Batched CG
                        P2, _ = batched_cg_conv(
                            X, P1,
                            module=module,
                            num_iters=num_iters,
                            eps=eps
                        )

                        # Post diagonal scaling to imitate -1/2
                        x_scaling = torch.sqrt(act_diag_conv(X, module=module))  # C_in x kh x kw
                        direction = P2.mul_(x_scaling.unsqueeze(0)).view_as(p)
                    else:
                        raise ValueError(f"Invalid module: {module}")
                else:
                    direction = eff_grad

                p.add_(direction, alpha=-lr)

        # Clear references
        self.in_acts.clear()
        self.out_grads.clear()
        return loss

    def close(self):
        """
        Remove all hooks at the end of training
        """

        for h in self.hooks:
            h.remove()
        self.hooks.clear()
