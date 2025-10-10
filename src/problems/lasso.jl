module ADMM_Lasso

using LinearAlgebra
using MPI
using Main.ADMM

export LassoProblem

"""
Define problem
"""

Base.@kwdef mutable struct LassoProblem
    A::AbstractMatrix{Float64} # design matrix (m x n)
    b::AbstractVector{Float64} # observations (m x 1)
    λ::Float64 = 0.1 # L1 penalty parameter
end

mutable struct LassoContext
    A′b::Vector{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    skinny::Bool
    Aq::Vector{Float64}
    p::Vector{Float64}
end

"""
Define traits
"""

# Distribute across multiple processes
ADMM.DistributionTrait(::Type{<:LassoProblem}) = ADMM.MPIConsensus()

# Closed-form proximal operator (soft-thresholding)
ADMM.ProximalTrait(::Type{<:LassoProblem}) = ADMM.ClosedFormProx()


"""
Custom functions
"""

function ADMM.setup!(state::ADMM.ADMMState{LassoProblem, C}) where C
    problem = state.problem
    A = problem.A
    b = problem.b
    ρ = state.params.ρ
    
    # Dims
    m, n = size(A)
    
    # Initialize ADMM variables
    x = zeros(n)
    u = zeros(n)
    z = zeros(n)
    z_prev = zeros(n)
    r = zeros(n)
    w = zeros(n)
    q = zeros(n)
    
    # Cache expensive computations
    A′b = A' * b
    skinny = (m >= n)
    
    if skinny
        M = A' * A
        for i in 1:n
            M[i,i] += ρ
        end
        L = cholesky(Symmetric(M, :L)).L
        ctx = LassoContext(A′b, L, skinny, zeros(m), zeros(m))
    else
        M = (1/ρ) * (A * A')
        for i in 1:m
            M[i,i] += 1.0
        end
        L = cholesky(Symmetric(M, :L)).L
        ctx = LassoContext(A′b, L, skinny, zeros(m), zeros(m))
    end
    
    # Return new properly-typed state
    return ADMM.ADMMState(
        problem, state.comm, state.rank, state.nprocs,
        m, n, x, u, z, z_prev, r, w, q, ctx, state.params
    )
end

# f(x) = (1/2)||Ax - b||²
function ADMM.evaluate_objective(problem::LassoProblem, x)
    residual = problem.A * x - problem.b
    return 0.5 * dot(residual, residual)
end

# g(z) = λ||z||₁
function ADMM.evaluate_global_regularizer(problem::LassoProblem, z)
    return problem.λ * norm(z, 1)
end

function ADMM._admm_rho_changed!(state::ADMM.ADMMState{LassoProblem, LassoContext})
    A = state.problem.A
    m, n = size(A)
    ρ = state.params.ρ
    ctx = state.ctx

    if ctx.skinny
        M = A' * A
        @inbounds @views for i in 1:n; M[i,i] += ρ; end
        ctx.L = cholesky!(Symmetric(M, :L)).L
    else
        M = A * A'
        LinearAlgebra.scale!(M, 1/ρ)
        @inbounds @views for i in 1:m; M[i,i] += 1.0; end
        ctx.L = cholesky!(Symmetric(M, :L)).L
    end
    return nothing
end

# closed form updates
function ADMM._x_update!(state::ADMM.ADMMState{LassoProblem, LassoContext})
    ρ = state.params.ρ
    ctx = state.ctx
    A = state.problem.A

    @. state.q = ρ * (state.z - state.u) + ctx.A′b

    if ctx.skinny
        # Solve (A'A + ρI)x = q using cached Cholesky
        # Fix: destination, matrix, source
        ldiv!(state.x, ctx.L, state.q)       # state.x = L \ q
        ldiv!(state.x, ctx.L', state.x)      # state.x = L' \ x
    else
        # Woodbury: x = q/ρ - (1/ρ²) A' (I + (1/ρ)AA')^(-1) (Aq)
        mul!(ctx.Aq, A, state.q)             # Aq = A * q
        ldiv!(ctx.p, ctx.L, ctx.Aq)          # ctx.p = L \ Aq
        ldiv!(ctx.p, ctx.L', ctx.p)          # ctx.p = L' \ p
        mul!(state.x, A', ctx.p)             # x = A' * p
        @. state.x = state.q/ρ - state.x/(ρ*ρ)
    end
end

function ADMM._apply_proximal!(state::ADMM.ADMMState{LassoProblem, LassoContext}, ::ADMM.ClosedFormProx)
    λ = state.problem.λ
    μ = state.nprocs * state.params.ρ  # Nρ
    τ = λ / μ # threshold
    
    # soft-thresholding
    @inbounds for i in eachindex(state.z)
        zi = state.z[i]
        if zi > τ
            state.z[i] = zi - τ
        elseif zi < -τ
            state.z[i] = zi + τ
        else
            state.z[i] = 0.0
        end
    end
end

function objective_value(state::ADMM.ADMMState{LassoProblem, LassoContext})
    problem = state.problem
    data_fit = ADMM.evaluate_objective(problem, state.x)
    regularizer = ADMM.evaluate_global_regularizer(problem, state.z)
    return data_fit + regularizer
end

end # module ADMM_Lasso