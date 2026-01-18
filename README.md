# FROG Optimizer

FROG (Fisher ROw-wise preconditioninG) is a K-FAC inspired optimizer with several structural modifications aimed at making second-order preconditioning practical and stable.

## Motivation

This optimizer is motivated by the following observations:

- Scale-invariance is a desirable property for the optimizer, as in Adam or RMSProp. Newtonian methods are not scale invariant and have scale as inverse gradient.

- At the later stages of training, decoupling X and Y seems to hurt performance.

- Y^T Y typically exhibits strong diagonal dominance. This suggests that Fisher may be close to block-diagonal w.r.t rows, making it effective to precondition rows (output channels) independently without huge loss in performance.

- Exact empirical Fisher construction is prohibitively expensive. So, subsampling and batched CG is used.


## Key modifications

- No decoupling of Fisher into kronecker product, as in K-FAC.

- Row-wise separation: each row has its own Fisher matrix.

- Rescaling Fisher by tr(F)/D (average diagonal value) to achieve scale-invariance.   

- Iterative CG on row-wise Fishers

- Subsampled activations
  Fisher matvecs are computed using a random subset of activations, significantly reducing computational cost.

## Legacy version: K-FROG (Kronecker FROG)
I kept the previous optimizer, which performs kronecker-based factoring, same as in K-FAC and uses CG on XX^T, while performing inverse sqrt on YY^T.

## TODO

- Add support for linear layers and biases.
- Add mixed precision support
- Compare FROG with other optimizers on convolutional architectures.
