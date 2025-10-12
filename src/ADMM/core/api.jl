# main API for engaging with ADMM

"""
_step!
- run a single step of ADMM optimizer
"""
function _step!(state::ADMMState)

    # store previous z state
    copyto!(state.z_prev, state.z)

    # update x
    _x_update!(state)

    # update z
    _z_update!(state)

    # check convergence
    residual_primal, residual_dual, epsilon_primal, epsilon_dual = check_convergence!(state)

    # adaptive μ adjustment
    maybe_adapt_mu!(state, residual_primal, residual_dual)

    # is it converged?
    converged = (residual_primal <= epsilon_primal) && (residual_dual <= epsilon_dual)

    return converged, residual_primal, residual_dual, epsilon_primal, epsilon_dual

end

"""
init
- initialize the ADMM optimizer with a custom problem type
- requires custom setup to fully complete initialization
"""
function init(problem; params=ADMMParams(), comm=nothing)
    dist_trait = DistributionTrait(typeof(problem))

    if !MPI.Initialized() # initialize MPI
        MPI.Init()
    end

    if dist_trait isa Serial
        rank = 0
        nprocs = 1
        comm = MPI.COMM_SELF
    else
        comm = something(comm, MPI.COMM_WORLD)  # use provided or default to COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
    end

    # Create state with Nothing context initially
    state = ADMMState(problem, comm, rank, nprocs,
                      0, 0, Float64[], Float64[], Float64[], Float64[], Float64[], 
                      Float64[], nothing, params)
    
    # setup! returns properly-typed state
    state = setup!(state)
    
    return state
end

"""
run_admm!
- run for max_iter with printouts every 10 iterations by default
"""
function run_admm!(state::ADMMState; max_iter::Int=1000, verbose::Bool=true)

    for iter in 1:max_iter
        converged, residual_primal, residual_dual, epsilon_primal, epsilon_dual = _step!(state)

        if verbose && (iter % 10 == 0 || iter <= 5)
            if state.rank == 0
                @printf("Iter %4d: primal=%.2e (tol=%.2e), dual=%.2e (tol=%.2e)\n",
                       iter, residual_primal, epsilon_primal, residual_dual, epsilon_dual)
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