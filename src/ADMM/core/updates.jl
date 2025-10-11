# functions for updating primal x and dual z variables
# can be optionally overwritten by user but recommend default

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
        @. state.z_work = state.x + state.u # macro @. just allows vector addition without having to specify
    else
        @. state.z_work = (params.α * state.x + (1 - params.α) * state.z_prev) + state.u
    end

    # average z_work across all ranks if using MPI
    _average_z!(state, DistributionTrait(typeof(state.problem)))

    _apply_proximal!(state, ProximalTrait(typeof(state.problem)))
end

"""
multiple dispatch helper functions
"""
function _average_z!(state::ADMMState, ::MPIConsensus)
    MPI.Allreduce!(state.z_work, state.z, MPI.SUM, state.comm) # reduction operation (here, MPI.SUM) across all ranks
    @. state.z = state.z / state.nprocs # get the average (divide SUM / N)
end

function _average_z!(state::ADMMState, ::Serial)
    copyto!(state.z, state.z_work)
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