using Pkg
Pkg.activate(".")

# Load modules
include("../src/admm.jl")
include("../src/problems/lasso.jl")

using Main.ADMM
using Main.ADMM_Lasso

using LinearAlgebra
using SparseArrays
using MPI

# Initialize MPI
MPI.Init()

# Create synthetic problem
println("Creating synthetic LASSO problem...")
m, n = 100, 50  # overdetermined system
A = randn(m, n)
x_true = sprandn(n, 0.1)  # 10% sparse
b = A * x_true + 0.01 * randn(m)

println("True solution sparsity: $(count(!iszero, x_true))/$n")

# Setup ADMM parameters
params = ADMMParams(
    ρ = 1.0,
    reltol = 1e-3,
    abstol = 1e-4,
    α = 1.0,
    adaptive_ρ = true
)

# Create problem and solve
println("\nSolving with ADMM...")
problem = LassoProblem(A=A, b=b, λ=0.1)
state = init(problem; params=params)
state, iters, converged = run_admm!(state; max_iter=100, verbose=true)

# Check solution
println("\n" * "="^50)
println("Results:")
println("="^50)
println("Converged: $converged in $iters iterations")
println("Solution sparsity: $(count(abs.(state.z) .> 1e-4))/$n")
println("Recovery error: $(norm(state.z - x_true) / norm(x_true))")
println("Objective value: $(ADMM_Lasso.objective_value(state))")

# Finalize MPI
#MPI.Finalize()