# FROG Optimizer

FROG (Fisher ROw-wise preconditioninG) is a K-FAC inspired optimizer with several structural modifications aimed at making second-order preconditioning practical and stable.

## Key modifications

- Row-wise output conditioning  
  The output-side Fisher factor (Y^T Y) is approximated diagonally, resulting in independent preconditioning for each output channel.

- Inverse square-root scaling  
  Uses the inverse square root of the diagonal Y^T Y instead of the full inverse.

- Iterative inversion of activation covariance  
  The input-side covariance (X^T X) is inverted approximately using an iterative Conjugate Gradient (CG) solver rather than exact inversion.

- Implicit covariance products  
  Products with X^T X are computed implicitly using convolution and conv2d_weight, without explicitly forming the covariance matrix.

- Subsampled activation statistics  
  Curvature statistics are computed using a random subset of activations, significantly reducing computational cost.

- Additional input-side scaling  
  Applies an extra scaling using the square root of the diagonal of X^T X.

## Motivation

These design choices are motivated by the following observations:

- Inverse square-root Fisher preconditioning often works better than a full inverse, as seen in optimizers such as RMSProp and Adam.

- Explicit construction of the activation covariance X^T X is prohibitively expensive for convolutional layers due to kernel expansion.

- Exact inversion of X^T X tends to amplify noise in low-eigenvalue directions; iterative solvers implicitly regularize these directions.

## TODO

- Add support for linear layers.
- Explore alternative activation subsampling strategies, including maintaining a pool of samples across optimization steps.
- Compare FROG with other optimizers on convolutional architectures.