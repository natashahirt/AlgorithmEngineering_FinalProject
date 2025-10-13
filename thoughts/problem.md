# Breaking Down the Stress-Constrained Topology Optimization Problem

## Key Concepts

### Separable Lagrangian

- Allows parallel updates for each element e since α, σ, and λ are independent
- "Local" update refers to element-wise independence, not spatial zones
- Each element updated independently of global optimization

### Core Challenge: Stress Constraints

- Constraint `σ_e(ρ) < σ_lim` for all elements e is highly nonlinear
- Stress represents internal force in elements
- Want to minimize stress to prevent material failure
- Traditional approaches struggle because:
  - Each element's stress connects to all other elements via displacement field U
  - U has nonlinear dependence on stiffness matrix K
  - Results in dense coupling and massive matrix problem

### Solution Approach

- Introduce auxiliary variable α to split stress constraint:
  1. `α_e ≈ σ_e(ρ)`: Couples actual stresses to auxiliary variable
  2. `α_e <= σ_lim`: Enforces stress limit
- Lagrange multiplier λ enables separate optimization by:
  - Adding penalty for mismatch between α_e and σ_e
  - Makes true stress σ converge to α
  - Stress bounds enforced without direct coupling to ρ

### Design Variables

- ϕ: Raw algorithmic variables (smooth/continuous)
- ρ: Physical densities used in FEM
  - Derived from ϕ through:
    1. Density filtering (stenciling)
    2. Heaviside projection (thresholding)
  - Dependent on stencil radius r
  - Results in binary values

### Stress Notation

- σ: Full stress tensor (normal + shear components)
- σ̄: von Mises stress (scalar equivalent)
- σ̃: "Relaxed stress"
  - Addresses near-void elements
  - Uses q-relaxation to prevent division-by-zero
  - Scales density by power q (e.g. 0.5)

## Algorithm Overview

### ADMM Outer Loop

1. Update design ρ (using α^{i-1}, λ^{i-1})
2. Update auxiliary α (using ρ^{i}, λ^{i-1})
3. Update lagrangian λ (using ρ^{i}, α^{i})

### Detailed Steps

#### 1. Design Update (Inner Loop - Global MMA)

- Filter ϕ → ρ
- Assemble K(ρ)
- Solve K(ρ)U = F for displacement
- Compute element stresses σ_e
- Calculate adjoint lagrangian gradient
- Update ϕ via MMA
- Repeat until converged

#### 2. Auxiliary Update

- Project local stress
- If α_e <= σ_lim: retain α_e
- Else: project to σ_lim
- Affects λ update to guide σ to feasible region

#### 3. Lagrangian Update

- Update λ per element
- Adjusts penalties to minimize α-σ gap
- Check ADMM convergence
- Tune stenciling/penalty parameters if needed
