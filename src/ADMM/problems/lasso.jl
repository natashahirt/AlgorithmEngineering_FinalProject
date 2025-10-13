export LassoProblem

"""
Define problem
"""

Base.@kwdef mutable struct LassoProblem{D <: ADMM.DistributionMode}
    A::AbstractMatrix{Float64} # design matrix (m x n)
    b::AbstractVector{Float64} # observations (m x 1)
    λ::Float64 = 0.1 # L1 penalty parameter
    distribution::D = ADMM.Serial() # default to serial
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
# ADMM.DistributionTrait(::Type{<:LassoProblem}) = ADMM.MPIConsensus()
ADMM.DistributionTrait(::Type{<:LassoProblem{D}}) where D = D()

# Closed-form proximal operator (soft-thresholding)
ADMM.ProximalTrait(::Type{<:LassoProblem}) = ADMM.ClosedFormProx()


"""
Custom functions
"""

function ADMM.setup!(state::ADMM.ADMMState{LassoProblem{D}, C}) where {D, C}
    problem = state.problem
    A = problem.A
    b = problem.b
    μ = state.params.μ
    
    # Dims
    m, n = size(A)
    
    # Initialize ADMM variables
    x = zeros(n)
    u = zeros(n)
    z = zeros(n)
    z_prev = zeros(n)
    primal_res = zeros(n)
    z_work = zeros(n)
    
    # Cache expensive computations
    A′b = A' * b
    skinny = (m >= n)
    
    if skinny
        M = A' * A
        for i in 1:n
            M[i,i] += μ
        end
        L = cholesky(Symmetric(M, :L)).L
        ctx = LassoContext(A′b, L, skinny, zeros(m), zeros(m))
    else
        M = (1/μ) * (A * A')
        for i in 1:m
            M[i,i] += 1.0
        end
        L = cholesky(Symmetric(M, :L)).L
        ctx = LassoContext(A′b, L, skinny, zeros(m), zeros(m))
    end
    
    # Return new properly-typed state
    return ADMM.ADMMState(
        problem, state.iter, state.comm, state.rank, state.nprocs,
        m, n, x, u, z, z_prev, primal_res, z_work, ctx, state.params
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

function ADMM._admm_mu_changed!(state::ADMM.ADMMState{LassoProblem{D}, LassoContext}) where D
    A = state.problem.A
    m, n = size(A)
    μ = state.params.μ
    ctx = state.ctx

    if ctx.skinny
        M = A' * A
        @inbounds @views for i in 1:n; M[i,i] += μ; end
        ctx.L = cholesky!(Symmetric(M, :L)).L
    else
        M = A * A'
        LinearAlgebra.scale!(M, 1/μ)
        @inbounds @views for i in 1:m; M[i,i] += 1.0; end
        ctx.L = cholesky!(Symmetric(M, :L)).L
    end
    return nothing
end

# closed form updates
function ADMM._x_update!(state::ADMM.ADMMState{LassoProblem{D}, LassoContext}) where D
    μ = state.params.μ
    ctx = state.ctx
    A = state.problem.A

    # Compute RHS vector (local variable, no need to store in state)
    rhs = μ * (state.z .- state.u) .+ ctx.A′b

    if ctx.skinny
        # Solve (A'A + μI)x = rhs using cached Cholesky
        # Fix: destination, matrix, source
        ldiv!(state.x, ctx.L, rhs)       # state.x = L \ rhs
        ldiv!(state.x, ctx.L', state.x)  # state.x = L' \ x
    else
        # Woodbury: x = rhs/μ - (1/μ²) A' (I + (1/μ)AA')^(-1) (A*rhs)
        mul!(ctx.Aq, A, rhs)             # Aq = A * rhs
        ldiv!(ctx.p, ctx.L, ctx.Aq)      # ctx.p = L \ Aq
        ldiv!(ctx.p, ctx.L', ctx.p)      # ctx.p = L' \ p
        mul!(state.x, A', ctx.p)         # x = A' * p
        @. state.x = rhs/μ - state.x/(μ*μ)
    end
    
    return nothing
end

function ADMM._apply_proximal!(state::ADMM.ADMMState{LassoProblem{D}, LassoContext}, ::ADMM.ClosedFormProx) where D
    λ = state.problem.λ
    penalty = state.nprocs * state.params.μ  # Nμ
    τ = λ / penalty # threshold
    
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
    
    return nothing
end