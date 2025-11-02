# stress related functions

"""
get the element stresses σ, as well as von mises stresses σ̄ and the relaxed stresses σ̃
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
        ρ_e = max(ctx.ρ[i], problem.ρ_min)
        ctx.σ̃[i] = ctx.σ̄[i] * (ρ_e^problem.q_relax)

    end
    
    return nothing

end

"""
von_mises_stress for 2d and 3d
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

"""
get the derivatives of the stresses
all arithmetic so can pre-designate the gradients
"""
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