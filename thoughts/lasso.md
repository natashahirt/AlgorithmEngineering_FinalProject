# Lasso problem

## Summary

LASSO = Least Absolute Shrinkage and Selection Operator.
Finds sparse solutions (with many zeros) to the problem "given A and noisy b, recover the sparse solution x" by solving the optimization problem:

minimize (1/2)||Ax - b||₂² + λ||x||₁

where:

- A is the feature/measurement matrix
- x is the variable we want to find (sparse solution)
- b is the observation/target vector
- λ is the regularization parameter controlling sparsity
- ||·||₁ denotes L1 norm
- ||·||₂² denotes squared L2 norm

This combines:

1. Least squares fit: (1/2)||Ax - b||₂² (data-fitting)
2. L1 regularization: λ||x||₁ to promote sparsity (becoming sparse)

The ADMM reformulation is essentially identical:

minimize (1/2)||Ax - b||₂² + λ||z||₁
s.t. x = z

This formulation splits the problem into two simpler sub-problems, namely a least-squares (quadratic) and soft-thresholding (L1 norm). It is therefore easier to solve iteratively. ADMM uses consensus to force the two variables x and z into agreement so that the data is fit and the result is sparse. If we were to use MPI, then for each subprocess xi we would have a layer of consensus with z.

We first try to solve the least squares problem using Cholesky factorization:

argmin_x (1/2)||Ax - b||² + (ρ/2)||x - (z - u)||²

Which is quite fast (only two triangular solves per iteration). This part of the problem finds x that fits data Ax ≈ b while staying close to z - u, where z is the current consensus.

Then we solve the L1 regularization part:

argmin_z λ||z||₁ + (ρ/2)||x + u - z||²

This has a closed-form solution using soft thresholding i.e. values are shrunk toward zero by an amount κ (here, {λ/ρ}. If a value is smaller than κ, it is zero, else it is reduced by κ while the sign is retained. This creates sparse solutions by eliminating small coefficients):

z = S_{λ/ρ}(x + u)

where S is the soft thresholding operator:

S_κ(a) = sign(a) * max(|a| - κ, 0)

The u update is a simple dual variable update:

u = u + (x - z)

This process repeats until convergence, with each iteration:

1. Solving least squares for x
2. Soft thresholding for z
3. Dual update for u
