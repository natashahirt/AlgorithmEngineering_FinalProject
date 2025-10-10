# virtual functions for problem implementation

function setup!(state::ADMMState)
    error("setup! is not implemented for problem $(typeof(state.problem))")
end

function objective_local(state::ADMMState, x)
    return evaluate_objective(state.problem, x)
end

function evaluate_objective(problem, x)
    error("evaluate_objective is not implemented for problem $(typeof(problem))")
end

# override with specific problems' g functions
# default is no regularization g(z)=0 to give simple averaging update (z = x̄ + ū)
function evaluate_global_regularizer(problem, z)
    println("using default evaluate_global_regularizer g(z)=0. If you need custom regularization, make sure you implement it.")
    return 0.0 
end

# default is no-op
# problem-specific implementations should rebuild ρ-dependent caches
function _admm_rho_changed!(state::ADMMState)
    println("using default _admm_rho_changed! (no-op). If caches need to be rebuilt, make sure you implement it.")
    return nothing
end

# default implementation for getting the objective value
function objective_value(state::ADMMState)
    return evaluate_objective(state.problem, state.x) + 
           evaluate_global_regularizer(state.problem, state.z)
end