using Pkg; Pkg.activate(dirname(@__DIR__)) # activate project root

# modules
using ADMM
using SparseArrays: sprandn
using LinearAlgebra: norm
using MPI

MPI.Init()

# synthetic problem
println("Creating synthetic LASSO problem...")
m, n = 100, 50  # overdetermined system
A = randn(m, n);
x_true = sprandn(n, 0.1);  # 10% sparse
b = A * x_true + 0.01 * randn(m);

println("True solution sparsity: $(count(!iszero, x_true))/$n")

# ADMM parameters
params = ADMMParams(
    ρ = 1.0,
    reltol = 1e-3,
    abstol = 1e-4,
    α = 1.0,
    adaptive_ρ = true
)

# solve problem
println("\nSolving with ADMM...")
problem = LassoProblem(A=A, b=b, λ=0.1, distribution=ADMM.Serial());
state = init(problem; params=params);
state, iters, converged = run_admm!(state; max_iter=100, verbose=true);

# check solution
println("\n" * "="^50)
println("Results:")
println("="^50)
println("Converged: $converged in $iters iterations")
println("Solution sparsity: $(count(abs.(state.z) .> 1e-4))/")
println("Recovery error: $(norm(state.z - x_true) / norm(x_true))")

obj_val = evaluate_objective(problem, state.x) + evaluate_global_regularizer(problem, state.z);
println("Objective value: $obj_val")

# Finalize MPI
MPI.Finalize()