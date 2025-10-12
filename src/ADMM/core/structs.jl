# primary structures

"""
ADMMParams
- controls parameters for optimizer (penalty, tolerances etc.)
"""
Base.@kwdef mutable struct ADMMParams # note: kwdef automatically makes all these vars keywords
    μ::Float64      = 1.0 # augmented lagrangian penalty param
    reltol::Float64 = 1e-2 # convergence tolerance (relative)
    abstol::Float64 = 1e-4 # convergence tolerance (absolute)
    α::Float64      = 1.0 # over-relaxation parameter (1.0 = off, >1 speeds up convergence)
    adaptive_μ::Bool = false # do we automatically adjust μ?
end

"""
ADMMState
- keeps track of state of given optimization
- accepts custom Problem and Context types (P, C)
"""
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
    primal_res::Vector{Float64} # local primal residual (how far is xᵢ from consensus z?)

    # Work buffers (temp)
    z_work::Vector{Float64} # working version of z before averaging/proximal (x + u)

    # Problem-specific cached context (e.g., Atb, Cholesky factor)
    ctx::C # avoid recomputing expensive operations e.g. least squares, lasso, logistic regression

    # Parameters
    params::ADMMParams # from struct above, used to tune ADMM alg
end