module ADMM

# import
using MPI # message passing interface for distributed systems
using LinearAlgebra
using Optim # for optimization
using Mooncake # automatic differentiation
using DifferentiationInterface

# export
export ADMMParams, ADMMState
export init, run_admm!, setup!
export evaluate_objective, evaluate_global_regularizer
export DistributionMode, MPIConsensus, Serial
export ProximalMode, ClosedFormProx, NumericalProx
export DistributionTrait, ProximalTrait
export _x_update!, _z_update!, _apply_proximal!, _admm_rho_changed!  # for advanced customization

""" =======
Traits
======= """

# Distribution modes (a trait)
# to be called down the line
abstract type DistributionMode end
struct MPIConsensus <: DistributionMode end
struct Serial       <: DistributionMode end

DistributionTrait(::Type) = MPIConsensus()

# Proximal operator mode trait
abstract type ProximalMode end
struct ClosedFormProx <: ProximalMode end # has closed-form solution (e.g., LASSO soft-threshold)
struct NumericalProx <: ProximalMode end # needs numerical optimization (e.g. Optim.jl)

# Default: no global regularizer (identity prox)
ProximalTrait(::Type) = ClosedFormProx()


""" =======
Structs
======= """

Base.@kwdef mutable struct ADMMParams # note: kwdef automatically makes all these vars keywords
    ρ::Float64      = 1.0 # augmented lagrangian param (i.e. a penalty)
    reltol::Float64 = 1e-2 # convergence tolerance (relative)
    abstol::Float64 = 1e-4 # convergence tolerance (absolute)
    α::Float64      = 1.0 # over-relaxation parameter (1.0 = off, >1 speeds up convergence)
    adaptive_ρ::Bool = false # do we automatically adjust ρ?
end

mutable struct ADMMState{P,C}
    problem::P # define own struct

    # MPI tools
    comm::MPI.Comm # MPI communicator object e.g. MPI.COMM_WORLD
    rank::Int # unique identifier for current process (which node am I)
    nprocs::Int # number of processes in the communicator (total nodes)

    # Dims 
    m::Int # local problem size (each rank/node has own m)
    n::Int # consensus variable (global n is shared, size of ADMM variables below)

    # ADMM variables (vectors, size n)
    x::Vector{Float64} # local primal xᵢ (rank's current estimate of solution)
    u::Vector{Float64} # local scaled dual uᵢ (lagrange multiplier for consensus constraint, makes sure xᵢ == z)
    z::Vector{Float64} # global consensus (all ranks must agree on this one, is an average across ranks)
    z_prev::Vector{Float64} # previous z (check for convergence and calculate dual residual)
    r::Vector{Float64} # local primal residual (how far is xᵢ from consensus z?)

    # Work buffers (temp)
    w::Vector{Float64} # temp storage for intermediate calculations (avoid memory allocation)
    q::Vector{Float64} # temp storage for intermediate calculations (avoid memory allocation)

    # Problem-specific cached context (e.g., Atb, Cholesky factor)
    ctx::C # avoid recomputing expensive operations e.g. least squares, lasso, logistic regression

    # Parameters
    params::ADMMParams # from struct above, used to tune ADMM alg
end


""" =======
ADMM functions
======= """

function setup!(state::ADMMState)
    error("setup")
end

function objective_local(state::ADMMState, x)
    return evaluate_objective(state.problem, x)
end

function evaluate_objective(problem, x)
    error("evaluate_objective not implemented for problem $(typeof(problem))")
end

# override with specific problems' g functions
# default is no regularization g(z)=0 to give simple averaging update (z = x̄ + ū)
function evaluate_global_regularizer(problem, z)
    return 0.0 
end

# default is no-op
# problem-specific implementations should rebuild ρ-dependent caches
function _admm_rho_changed!(state::ADMMState)
    return nothing
end

"""
default _x_update!(state::ADMMState)
for custom, override, e.g. _x_update!(state::ADMMState{LeastSquaresProblem})
"""
function _x_update!(state::ADMMState)
# solve argmin_x f_i(x) + (ρ/2)||x - (z - u)||²
# write into state.x
    
    params = state.params
    ρ = params.ρ

    # get the target: x should approximate (z - u)
    target = state.z .- state.u

    # get augmented lagrangian
    # f_i(x) + (ρ/2)||x - (z - u)||²
    function augmented_lagrangian(x)
        f_val = objective_local(state, x)
        quad_penalty = (ρ/2) * norm(x - target)^2
        return f_val + quad_penalty
    end

    backend = AutoMooncake()
    grad_func = prepare_gradient(augmented_lagrangian, backend, state.x)

    # solve using Optim.jl
    result = optimize(augmented_lagrangian, (g, x) -> gradient!(g, grad_func, backend, x), state.x,
                      method=LBFGS(), show_trace=false)

    state.x .= Optim.minimizer(result)

end

"""
default _z_update!(state::ADMMState)
for custom, override, e.g. _z_update!(state::ADMMState{LassoProblem})
"""
function _z_update!(state::ADMMState)
# update the global z (write into state.z)
    params = state.params

    # over-relax using α
    # allows ADMM to take larger steps in the update direction
    if params.α == 1.0
        @. state.w = state.x + state.u # macro @. just allows vector addition without having to specify
    else
        @. state.w = (params.α * state.x + (1 - params.α) * state.z_prev) + state.u
    end

    # average w across all z if using MPI
    _average_z!(state, DistributionTrait(typeof(state.problem)))

    _apply_proximal!(state, ProximalTrait(typeof(state.problem)))
end

function _average_z!(state::ADMMState, ::MPIConsensus)
    MPI.Allreduce!(state.w, state.z, MPI.SUM, state.comm) # reduction operation (here, MPI.SUM) across all ranks
    @. state.z = state.z / state.nprocs # get the average (divide SUM / N)
end

function _average_z!(state::ADMMState, ::Serial)
    copyto!(state.z, state.w)
end

function _apply_proximal!(state::ADMMState, ::ClosedFormProx)
    # z stays as average, no regularization
end

function _apply_proximal!(state::ADMMState, ::NumericalProx)
    # numerical optimization for non-standard regularizers 
    # solve: z = prox_{g, Np}(x̄ + ū)
    # same as argmin_z { g(z) + (Nρ/2)||z - (x̄ + ū)||² }

    μ = state.nprocs * state.params.ρ # Nρ
    center = state.z

    # Default prox: identity (g ≡ 0). If a problem overrides evaluate_global_regularizer,
    # the generic proximal solve uses Optim; otherwise, it's a no-op.
    function proximal_objective(z)
        g_val = evaluate_global_regularizer(state.problem, z)
        # center is state.z already
        return g_val + (μ/2) * norm(z .- center)^2
    end

    result = optimize(proximal_objective, state.z,
                        method=LBFGS(), show_trace=false)

    state.z .= Optim.minimizer(result)
end

"""
check_convergence!(state::ADMMState)
"""
function check_convergence!(state::ADMMState)

    params = state.params
    ρ = params.ρ

    # get residuals
    @. state.r = state.x - state.z
    r2_local = dot(state.r, state.r)
    x2_local = dot(state.x, state.x)
    u2_local = dot(state.u, state.u)

    temp = [r2_local, x2_local, u2_local]
    _allreduce_inplace!(temp, DistributionTrait(typeof(state.problem)), state.comm)

    # norms used to calculate convergence
    residual_primal = sqrt(temp[1])     # ||r|| stack, norm of difference between x and z
    primal_norm = sqrt(temp[2])       # ||x|| stack, norm of primal variable x
    dual_norm = ρ * sqrt(temp[3])     # ||y||stack where y=ρu

    # dual residual: s = ρ √N ||z - z_prev||  (compute locally; z is identical on all ranks)
    z_diff2 = dot(state.z .- state.z_prev, state.z .- state.z_prev)
    residual_dual = ρ * sqrt(state.nprocs) * sqrt(z_diff2)

    # tolerances (from Boyd et al.)
    z2_local = dot(state.z, state.z)
    z_norm = _compute_z_norm(z2_local, DistributionTrait(typeof(state.problem)), state.comm, state.nprocs)

    sqrt_nN = sqrt(state.n * state.nprocs)
    epsilon_primal = sqrt_nN * params.abstol + params.reltol * max(primal_norm, z_norm)
    epsilon_dual = sqrt_nN * params.abstol + params.reltol * dual_norm

    # update u
    @. state.u = state.u + (state.x - state.z)

    return residual_primal, residual_dual, epsilon_primal, epsilon_dual

end

function maybe_adapt_rho!(state::ADMMState, primal_residual::Float64, dual_residual::Float64;
                          τ_incr::Float64=2.0, τ_decr::Float64=2.0,
                          rho_min::Float64=1e-6, rho_max::Float64=1e6)
    p = state.params
    p.adaptive_ρ || return  # skip
    
    ρ_old = p.ρ # store

    # increase ρ if primal_residual >> dual_residual
    if primal_residual > 10 * dual_residual && ρ_old < rho_max
        p.ρ = min(rho_max, ρ_old * τ_incr)
        @. state.u = state.u / τ_incr    # keep y = ρu invariant
    # decrease ρ if primal_residual << dual_residual
    elseif dual_residual > 10 * primal_residual && ρ_old > rho_min
        p.ρ = max(rho_min, ρ_old / τ_decr)
        @. state.u = state.u * τ_decr
    end
    
    # notify problem to rebuild caches
    if p.ρ != ρ_old
        if hasmethod(_admm_rho_changed!, Tuple{typeof(state)})
            _admm_rho_changed!(state)
        end
    end
end

function _allreduce_inplace!(data::Vector, ::MPIConsensus, comm::MPI.Comm)
    MPI.Allreduce!(data, MPI.SUM, comm)
end

function _allreduce_inplace!(data::Vector, ::Serial, comm::MPI.Comm)
    # No-op for serial
end

function _compute_z_norm(z2_local::Float64, ::MPIConsensus, comm::MPI.Comm, nprocs::Int)
    temp = [z2_local]
    MPI.Allreduce!(temp, MPI.SUM, comm)
    return sqrt(temp[1])  # √(N * ||z||²)
end

function _compute_z_norm(z2_local::Float64, ::Serial, comm::MPI.Comm, nprocs::Int)
    return sqrt(z2_local)
end

""" =======
ADMM step
======= """

function _step!(state::ADMMState)

    # store previous z state
    copyto!(state.z_prev, state.z)

    # update x
    _x_update!(state)

    # update z
    _z_update!(state)

    # check convergence
    residual_primal, residual_dual, epsilon_primal, epsilon_dual = check_convergence!(state)

    # adaptive ρ adjustment
    maybe_adapt_rho!(state, residual_primal, residual_dual)

    # is it converged?
    converged = (residual_primal <= epsilon_primal) && (residual_dual <= epsilon_dual)

    return converged, residual_primal, residual_dual, epsilon_primal, epsilon_dual

end

""" =======
run ADMM / API
======= """

function init(problem; params=ADMMParams(), comm=MPI.COMM_WORLD)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Create state with Nothing context initially
    state = ADMMState(problem, comm, rank, nprocs,
                      0, 0, Float64[], Float64[], Float64[], Float64[], Float64[], 
                      Float64[], Float64[], nothing, params)
    
    # setup! returns properly-typed state
    state = setup!(state)
    
    return state
end

function run_admm!(state::ADMMState; max_iter::Int=1000, verbose::Bool=true)

    for iter in 1:max_iter
        converged, residual_primal, residual_dual, epsilon_primal, epsilon_dual = _step!(state)

        if verbose && (iter % 10 == 0 || iter <= 5)
            if state.rank == 0
                println("Iter $iter: primal=$residual_primal (tol=$epsilon_primal), dual=$residual_dual (tol=$epsilon_dual)")
            end
        end

        if converged
            if state.rank == 0
                println("ADMM converged after $iter iterations")
            end
            return state, iter, true
        end
    end

    if state.rank == 0
        println("ADMM did not converge after $max_iter iterations")
    end

    return state, max_iter, false

end

end # of module