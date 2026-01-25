# FROG Optimizer

**FROG (Fisher ROw-wise preconditioninG)** is a second-order optimizer that approximates natural-gradient updates using row-wise Fisher preconditioning and batched Conjugate Gradient (CG) solves. 
It is inspired by K-FAC, but avoids Kronecker factorization and instead approximates the Fisher matrix as row-wise block-diagonal, using a small subset of samples to obtain stable, scale-free updates with low computational overhead.

A detailed technical description of the method is provided in **`technical_overview.pdf`**.

## Motivation

FROG is motivated by the following observations:

- Scale-invariance is a desirable optimizer property, as in Adam or RMSProp.  
  Classical Newton and natural-gradient methods are not scale-invariant and are sensitive to gradient magnitude.

- At later stages of training, decoupling input and output factors (X and Y), as done in K-FAC, often degrades performance.

- The empirical Fisher term \( Y^\top Y \) is typically strongly diagonally dominant, suggesting that cross-row correlations are weak and that row-wise preconditioning is a reasonable approximation.

- Exact empirical Fisher construction is prohibitively expensive; therefore subsampling and iterative solvers are required in practice.

## Method Overview

FROG approximates normalized natural-gradient updates by solving a **row-wise Fisher system** for each output row (or convolutional channel).  
Each row’s Fisher matrix is estimated from a small subset of activations and inverted approximately using a small number of batched CG iterations.

To stabilize learning and remove sensitivity to loss scaling, the Fisher matrix is normalized by its average diagonal value `tr(F) / D` and damped.  
Momentum is applied before Fisher preconditioning, and weight decay is handled in a decoupled manner.

## Key Modifications (relative to K-FAC)

- **No Kronecker factorization** of the Fisher matrix.

- **Row-wise Fisher separation**: each output row (or channel) is preconditioned independently.

- **Trace-normalized Fisher** `F / (tr(F) / D)` for scale-free updates.

- **Iterative Conjugate Gradient** instead of explicit matrix inversion.

- **Subsampled activations** for Fisher estimation, significantly reducing computational cost.

For full details, algorithmic description, and practical guidelines, see **`technical_overview.pdf`**.


## Wall-clock Efficiency (CIFAR-10)

All experiments were conducted on NVIDIA A100 (80 GB) using the CIFAR-10 dataset and the ResNet-18 model.  
Reported values correspond to wall-clock time (in minutes) required to first reach a given test accuracy.
A reference reproduction notebook is provided at `experiments/exp-cifar10.ipynb`.

The learning-rate schedule follows the setup from  
https://github.com/hirotomusiker/cifar10_pytorch/:  
a constant learning rate with 10× drops at epochs 80 and 120.
### Time to Reach Target Test Accuracy

| Optimizer | Batch Size (bs) | Sample Size (s) | 88% (min) | 90% (min) | 92% (min) | 94% (min) |
|-----------|----------------:|----------------:|----------:|----------:|----------:|----------:|
| FROG      | 512 | 64 | **1.29** | **1.86** | **5.37** | 11.88 |
| SGD       | 128 | – | 3.82 | 5.01 | 9.63 | **11.44** |
| SGD       | 512 | – | 3.91 | 7.80 | 8.84 | – |

### Training Dynamics (CIFAR-10)

<p align="center">
  <img src="figures/cifar10-loss.svg" alt="CIFAR-10 Training Loss vs Wall-Clock Time" width="48%" />
  <img src="figures/cifar10-acc.svg"  alt="CIFAR-10 Test Accuracy vs Wall-Clock Time" width="48%" />
</p>

### Interpretation

- FROG reaches 88–92% test accuracy between ~1.6× and ~4.2× faster in wall-clock time compared to SGD baselines.
- Final accuracy is comparable across methods; the main benefit is faster time-to-solution, especially in early and mid-training.


## Legacy version: K-FROG (Kronecker FROG)
I kept the previous optimizer, which performs kronecker-based factoring, same as in K-FAC and uses CG on XX^T, while performing inverse sqrt on YY^T.

## TODO

- Add support for linear layers and biases.
- Add mixed precision support
- Compare FROG with other optimizers on convolutional architectures.
