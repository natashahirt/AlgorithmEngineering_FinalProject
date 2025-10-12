# utility functions for topopt_continuous.jl

"""
build_filter_matrix(problem)

Build sparse filter matrix H and normalization vector Hs.
Uses KDTree for efficient neighbor search within filter radius.
"""
function build_filter_matrix(problem::TopOptProblem{D,T}) where {D,T}

    mesh = problem.mesh
    r_filter = problem.r_filter
    nel = length(mesh.elements)
    T_float = eltype(mesh.nodes[1].coords)

    centroids = FEM.get_mesh_centroids(mesh) # dim × nel
    volumes = FEM.get_mesh_volumes(mesh) # 1 × nel

    # nearest neighbors is a little elaborate when we have a pre-set grid
    # but useful if there's a more varied case
    tree = KDTree(centroids) # built in nearest neighbors
    
    # find the neighbors that lie within r_filter
    source_idx = Int[]
    neighbor_idx = Int[]
    weights = T_float[]

    for e in 1:nel

        neighbors = inrange(tree, centroids[e], r_filter)

        for i in neighbors

            dist = norm(centroids[e] - centroids[i])
            weight = (r_filter - dist) * volumes[i]

            push!(source_idx, e)
            push!(neighbor_idx, i)
            push!(weights, weight)

        end

    end

    H = sparse(source_idx, neighbor_idx, weights, nel, nel)
    Hs = vec(sum(H, dims=2))

    return H, Hs

end

function heaviside_projection(problem::TopOptProblem{D,T}, ϕ::T) where {D,T <: AbstractFloat}

    β = problem.β_heaviside # sharpness (larger = more binary)
    η = problem.η_heaviside # threshold (generally 0.5)

    tanh_βη = tanh(β * η)

    ρ = (tanh_βη + tanh(β * (ϕ - η))) / (tanh_βη + tanh(β * (one(T) - η)))

    return ρ

end

"""
apply_density_filter!(state)

apply density filter and Heaviside projection: ϕ → ρ (in place)
"""
function apply_density_filter!(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}

    problem = state.problem
    ctx = state.ctx

    # linear filter (spatial smoothing)
    mul!(ctx.ρ, ctx.H, ctx.ϕ) # ctx.ρ = H * ϕ
    ctx.ρ ./= ctx.Hs # element-wise division

    # heaviside projection (thresholding)
    @inbounds for i in eachindex(ctx.ρ)
        ctx.ρ[i] = heaviside_projection(problem, ctx.ρ[i])
    end

    # clamp to bounds
    clamp!(ctx.ρ, problem.ρ_min, problem.ρ_max)

    return nothing
end

"""
compute_element_stresses!(state)
"""
function compute_element_stresses!(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}

    problem = state.problem
    ctx = state.ctx
    mesh = problem.mesh

    for (i, element) in enumerate(mesh.elements)

        element_dofs = FEM.get_element_dofs(mesh, element)
        u_element = ctx.U[element_dofs]

        # σ_e = D_0 * B_c * u_e
        # → Voigt notation of stress tensor = elasticity tensor * strain/displacement matrix of centroid * displacement vector of element
        FEM.compute_element_strain!(ctx.ϵ_buffer, mesh, element, u_element, problem.analysis_type)

        D_const = FEM.get_constitutive_matrix(problem.material, problem.analysis_type)
        mul!(view(ctx.σ, :, i), D_const, vec(ctx.ϵ_buffer))

        # σ̄_e = (σ_e' * V * σ_e)^1/2
        # → von Mises stress of element = sqrt(stress * symmetric matrix V * stress)
        ctx.σ̄[i] = von_mises_stress(view(ctx.σ, :, i), problem.analysis_type)

        # σ̃_e = ρ_e ^ q * σ̄_e
        ctx.σ̃[i] = ctx.ρ[i]^problem.q_relax * ctx.σ̄[i]

    end

end

"""
von_mises_stress(σ, analysis_type)
"""
# for 2d
function von_mises_stress(σ::AbstractVector{T}, ::Union{FEM.PlaneStress, FEM.PlaneStrain}) where {T}
    # σ = [σxx, σyy, τxy]
    σxx = σ[1]
    σyy = σ[2]
    τxy = σ[3]
    
    # sqrt(σxx² - σxx*σyy + σyy² + 3*τxy²)
    return sqrt(σxx^2 - σxx*σyy + σyy^2 + 3*τxy^2)
end

# for 3d
function von_mises_stress(σ::AbstractVector{T}, ::FEM.ThreeDimensional) where {T}
    # σ = [σxx, σyy, σzz, τxy, τxz, τyz]
    σxx = σ[1]
    σyy = σ[2]
    σzz = σ[3]
    τxy = σ[4]
    τxz = σ[5]
    τyz = σ[6]
    
    # sqrt(0.5*((σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)² + 6*(τxy² + τxz² + τyz²)))
    return sqrt(0.5 * ((σxx-σyy)^2 + (σyy-σzz)^2 + (σzz-σxx)^2 + 6*(τxy^2 + τxz^2 + τyz^2)))
end

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
        weight = ctx.λ[i] + μ * (ctx.α[i] - ctx.σ̃[i]) # λᵢ + μ(αᵢ - σ̃ᵢ); not it's relaxed stress not von Mises

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

        # stress constraint derivative
        # (λᵢ + μ(αᵢ - σ̃ᵢ)) · q · ρᵢ^(q-1) · σ̃ᵢ
        weight = ctx.λ[i] + μ * (ctx.α[i] - ctx.σ̃[i])
        σ_constraint = weight * q * ctx.ρ[i]^(q-1) * ctx.σ̃[i]

        ctx.∇L_ϕ[i] = compliance_derivative + σ_constraint

    end

    # apply filter transpose
    # ∂L/∂ρ
    ctx.∇L_ϕ = ctx.H' * (ctx.∇L_ϕ ./ ctx.Hs)

    return nothing

end

# derivative varies based on 2d or 3d
# this is 2d
function ∂_von_mises_stress(σ::AbstractVector{T}, σ̄::T, ::Union{FEM.PlaneStress, FEM.PlaneStrain}) where {T}
    # σ = [σxx, σyy, τxy]
    # von Mises: σ̄ = √(σxx² - σxx*σyy + σyy² + 3*τxy²)
    # chain rule on sum of squared differences
    
    σxx = σ[1]
    σyy = σ[2]
    τxy = σ[3]
    
    # Avoid division by zero for negligible stress
    if σ̄ < eps(T)
        return zeros(T, 3, 1)
    end
    
    result = zeros(T, 3, 1)
    result[1] = (2*σxx - σyy) / (2*σ̄)
    result[2] = (2*σyy - σxx) / (2*σ̄)
    result[3] = 6*τxy / (2*σ̄)
    
    return result
end

# 3d
function ∂_von_mises_stress(σ::AbstractVector{T}, σ̄::T, ::FEM.ThreeDimensional) where {T}
    # σ = [σxx, σyy, σzz, τxy, τxz, τyz]
    # von Mises: σ̄ = √(0.5*((σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)² + 6*(τxy² + τxz² + τyz²)))
    # chain rule on the sum of squared differences
    
    σxx = σ[1]
    σyy = σ[2]
    σzz = σ[3]
    τxy = σ[4]
    τxz = σ[5]
    τyz = σ[6]
    
    if σ̄ < eps(T)
        return zeros(T, 6, 1)
    end
    
    # Derivatives (from chain rule)
    result = zeros(T, 6, 1)
    result[1] = 0.5 * (2*(σxx - σyy) - 2*(σzz - σxx)) / σ̄  # ∂σ̄/∂σxx = (2σxx - σyy - σzz)/σ̄
    result[2] = 0.5 * (2*(σyy - σzz) - 2*(σxx - σyy)) / σ̄  # ∂σ̄/∂σyy = (2σyy - σxx - σzz)/σ̄
    result[3] = 0.5 * (2*(σzz - σxx) - 2*(σyy - σzz)) / σ̄  # ∂σ̄/∂σzz = (2σzz - σxx - σyy)/σ̄
    result[4] = 6*τxy / σ̄  # ∂σ̄/∂τxy
    result[5] = 6*τxz / σ̄  # ∂σ̄/∂τxz
    result[6] = 6*τyz / σ̄  # ∂σ̄/∂τyz
    
    return result
end