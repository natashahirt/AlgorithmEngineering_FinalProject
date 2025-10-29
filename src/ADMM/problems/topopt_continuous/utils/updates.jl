"""
ADMM penalty parameter controlling tradeoff between primal and dual residuals
"""
function adapt_μ_topopt!(
    state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}, 
    primal_residual::T, 
    dual_residual::T;
    τ_incr::T=T(2.0), 
    τ_decr::T=T(2.0),
    μ_min::T=T(1e-4),
    μ_max::T=T(1e4)
) where {D,T}
    μ = state.params.μ

    if !state.params.adaptive_μ
        return nothing
    end

    if dual_residual < T(1e-12)  # z didn't move; skip μ update this iter
        return nothing
    end

    if primal_residual > T(10) * dual_residual
        state.params.μ = min(T(2) * μ, μ_max)
    elseif dual_residual > T(10) * primal_residual
        state.params.μ = max(μ / T(2), μ_min)
    end

    return nothing

end

"""
dropoff function for pushing values toward 0 or 1 in the topology optimization
higher β = sharper
schedule to gradually increase over the course of the optimization
"""
function update_heaviside_sharpness!(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}

    problem = state.problem

    # already at max sharpness
    if problem.β_heaviside >= problem.β_heaviside_max
        return nothing
    end

    ctx = state.ctx
    nel = ctx.nel

    # if heaviside schedule is enabled
    if problem.use_heaviside_schedule
        if mod(state.iter, problem.heaviside_schedule_frequency) == 0
            new_heaviside = compute_heaviside_schedule_value(problem, state.iter)
            problem.β_heaviside = min(new_heaviside, problem.β_heaviside_max)
        end
        return nothing
    end

    # Original adaptive schedule based on grey fraction
    grey_count = count(ctx.ρ) do ρ_val
        problem.grey_band_lo < ρ_val < problem.grey_band_hi
    end

    grey_fraction = nel == 0 ? zero(T) : T(grey_count) / T(nel)

    schedule_hit = problem.β_update_frequency > 0 && mod(state.iter, problem.β_update_frequency) == 0
    exceeds_grey = grey_fraction > problem.grey_fraction_trigger

    if !(schedule_hit || exceeds_grey)
        return nothing
    end

    new_heaviside = min(problem.β_heaviside * problem.β_heaviside_growth, problem.β_heaviside_max)

    if new_heaviside > problem.β_heaviside
        problem.β_heaviside = new_heaviside
    end

    return nothing
end

function compute_heaviside_schedule_value(problem::TopOptProblem{D,T}, iter::Int) where {D,T}
    if problem.heaviside_schedule_type == :exponential
        return problem.heaviside_schedule_start * (problem.heaviside_schedule_growth ^ iter)
    elseif problem.heaviside_schedule_type == :linear
        max_iter_est = 100  # reasonable default
        progress = min(iter / max_iter_est, 1.0)
        return problem.heaviside_schedule_start + (problem.heaviside_schedule_end - problem.heaviside_schedule_start) * progress
    elseif problem.heaviside_schedule_type == :step
        steps = iter ÷ problem.heaviside_schedule_frequency
        return problem.heaviside_schedule_start * (problem.heaviside_schedule_growth ^ steps)
    else
        error("Unknown heaviside schedule type: $(problem.heaviside_schedule_type)")
    end
end