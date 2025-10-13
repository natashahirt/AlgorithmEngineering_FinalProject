# get gradients for topopt_continuous.jl

"""
get the adjoint gradients
solution is found in the same way as Ku=f is found, except we're doing the whole thing on the
dual/lagrangian problem rather than the primal problem
"""
function compute_gradients_adjoint!(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}
    # after Zhai et al.

    problem = state.problem
    ctx = state.ctx
    mesh = problem.mesh

    μ = state.params.μ
    q = problem.q_relax
    E_0 = problem.material.E
    E_min = problem.ρ_min * E_0
    ρ_simp = problem.ρ_simp

    ndof = length(ctx.U)

    # right hand side (A1.14)
    # Σ Bᵢ^T · D₀^T ⋅ (∂σ̄ᵢ/∂σᵢ) ⋅ (λᵢ + μ(αᵢ - σ̃ᵢ)) · ρᵢ^q
    rhs = zeros(T, ndof)

    for (i, element) in enumerate(mesh.elements)

        element_dofs = FEM.get_element_dofs(mesh, element)

        # get B matrix (strain-displacement matrix) and D matrix (constitutive matrix)
        B = FEM.compute_B_matrix(mesh, element, problem.analysis_type)
        D_const = FEM.get_constitutive_matrix(problem.material, problem.analysis_type)

        # ∂σ̄/∂σ --derivative of von mises
        ∂σ̄_∂σ = ∂_von_mises_stress(view(ctx.σ, :, i), ctx.σ̄[i], problem.analysis_type) # ∂σ̄ᵢ/∂σᵢ

        # weight term
        weight = ctx.λ[i] + μ * (ctx.α[i] - ctx.σ̃[i]) # λᵢ + μ(αᵢ - σ̃ᵢ); note it's relaxed stress not von Mises

        # contribute element contribution
        dof_contribution = B' * D_const' * ∂σ̄_∂σ * (weight * ctx.ρ[i]^q) # full Bᵢ^T · D₀^T ⋅ (∂σ̄ᵢ/∂σᵢ) ⋅ (λᵢ + μ(αᵢ - σ̃ᵢ)) · ρᵢ^q

        # add to rhs
        for (local_i, global_i) in enumerate(element_dofs)
            rhs[global_i] += dof_contribution[local_i]
        end

    end

    # solve adjoint (dual!) system
    # K ⋅ λ_adjoint = RHS is equivalent to K ⋅ u = f
    ctx.λ_adjoint = FEM.solve_fem(ctx.K, rhs, problem.boundary_dofs)

    # get gradients ∂L/∂ρ
    # Θᵢ = λ_adjointᵢ^T · (∂Kᵢ/∂ρᵢ) · uᵢ + (λᵢ + μ(αᵢ - σ̃ᵢ)) · q · ρᵢ^(q-1) · σ̃ᵢ
    for (i, element) in enumerate(mesh.elements)

        element_dofs = FEM.get_element_dofs(mesh, element)
        u_element = ctx.U[element_dofs]
        λ_adjoint_element = ctx.λ_adjoint[element_dofs]

        # K_0 is base stiffness (no density scaling)
        K_0 = FEM.compute_element_stiffness(mesh, element, problem.material, problem.analysis_type)

        # compliance derivative λ_adjointᵢ^T · (∂Kᵢ/ρᵢ) · uᵢ
        # ∂Kᵢ/ρᵢ = ρ_simp · ρᵢ^(ρ_simp-1) · (E_0 - E_min) · K_0
        ∂K_∂ρ = ρ_simp * ctx.ρ[i]^(ρ_simp - 1) * (E_0 - E_min)
        compliance_derivative = ∂K_∂ρ * dot(λ_adjoint_element, K_0 * u_element)

        # pure compliance gradient (always present)
        pure_compliance = -∂K_∂ρ * dot(u_element, K_0 * u_element)

        # stress constraint derivative
        # (λᵢ + μ(αᵢ - σ̃ᵢ)) · q · ρᵢ^(q-1) · σ̃ᵢ
        weight = ctx.λ[i] + μ * (ctx.α[i] - ctx.σ̃[i])
        σ_constraint = weight * q * ctx.ρ[i]^(q-1) * ctx.σ̃[i]

        ctx.∇L_ϕ[i] = pure_compliance + compliance_derivative + σ_constraint

    end

    # apply filter transpose
    # ∂L/∂ρ
    ctx.∇L_ϕ = ctx.H' * (ctx.∇L_ϕ ./ ctx.Hs)

    return nothing

end