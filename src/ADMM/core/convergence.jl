# check whether optimization has converged

"""
check_convergence!(state::ADMMState)
"""
function check_convergence!(state::ADMMState)

    params = state.params
    ρ = params.ρ

    # get residuals
    @. state.primal_res = state.x - state.z
    r2_local = dot(state.primal_res, state.primal_res)
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

"""
multiple dispatch helper functions
"""
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

"""
maybe_adapt_rho!
- there may be situations where the optimization requires ρcontrolling strength of augmented Lagrangian) to increase or decrease from 
the default.

- increase: prioritize consensus (local variables pushed closer to the global avg)
- decrease: prioritize local objective optimization

N.B. when maybe_adapt_rho is called we need to notify the problem and rebuild caches
e.g. matrix factorizations that included ρ are no longer valid. The user must define
"""
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