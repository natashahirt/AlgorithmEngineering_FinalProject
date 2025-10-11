# materials.jl - material properties
"""
Material properties and constitutive models.
"""

"""
Material

Abstract type for material models.
"""
abstract type Material end

"""
LinearElastic{T}

Linear elastic material model.
"""
struct LinearElastic{T} <: Material
    E::T  # Young's modulus
    ν::T  # Poisson's ratio
    ρ::T  # Density
end

LinearElastic(E, ν) = LinearElastic(E, ν, 0.0)

"""
Analysis Types
"""
struct PlaneStress end
struct PlaneStrain end
struct ThreeDimensional end

"""
get_constitutive_matrix(material::LinearElastic, analysis_type)

Get constitutive matrix for linear elastic material.
"""
function get_constitutive_matrix(material::LinearElastic{T}, ::PlaneStress) where T
    E, ν = material.E, material.ν
    factor = E / (1 - ν^2)
    
    D = factor * [
        1    ν    0
        ν    1    0  
        0    0    (1-ν)/2
    ]
    
    return D
end

function get_constitutive_matrix(material::LinearElastic{T}, ::PlaneStrain) where T
    E, ν = material.E, material.ν
    factor = E / ((1 + ν) * (1 - 2ν))
    
    D = factor * [
        (1-ν)    ν       0
        ν        (1-ν)   0
        0        0       (1-2ν)/2
    ]
    
    return D
end

function get_constitutive_matrix(material::LinearElastic{T}, ::ThreeDimensional) where T
    E, ν = material.E, material.ν
    factor = E / ((1 + ν) * (1 - 2ν))
    
    D = factor * [
        (1-ν)    ν       ν       0       0       0
        ν        (1-ν)   ν       0       0       0
        ν        ν       (1-ν)   0       0       0
        0        0       0       (1-2ν)/2 0       0
        0        0       0       0       (1-2ν)/2 0
        0        0       0       0       0       (1-2ν)/2
    ]
    
    return D
end

"""
get_material_properties(material::Material)

Get material properties as named tuple.
"""
function get_material_properties(material::LinearElastic)
    return (E = material.E, ν = material.ν, ρ = material.ρ)
end

# =============================================================================
# Nonlinear Material Models
# =============================================================================

"""
NonlinearMaterial

Abstract type for nonlinear material models that use deformation gradient formulation.
"""
abstract type NonlinearMaterial <: Material end

"""
NeoHookean{T}

Neo-Hookean hyperelastic material model for large deformations.
Uses strain energy density: W = 0.5*μ*(I₁-3-2log(J)) + 0.5*λ*log(J)²
where I₁ = tr(F'F), J = det(F).
"""
struct NeoHookean{T} <: NonlinearMaterial
    E::T   # Young's modulus
    ν::T   # Poisson's ratio
    μ::T   # First Lamé parameter (shear modulus)
    λ::T   # Second Lamé parameter
    ρ::T   # Density
end

function NeoHookean(E::T, ν::T, ρ::T = zero(T)) where T
    μ = E / (2 * (1 + ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    NeoHookean(E, ν, μ, λ, ρ)
end

"""
energy_density(material::NeoHookean, F)

Compute strain energy density W for Neo-Hookean material.
W = 0.5*μ*(I₁ - dim - 2log(J)) + 0.5*λ*log(J)²
"""
function energy_density(material::NeoHookean, F::AbstractMatrix)
    dim = size(F, 1)
    I1 = dot(F, F)  # tr(F'*F)
    J = det(F)
    logJ = log(J)
    return 0.5 * material.μ * (I1 - dim - 2 * logJ) + 0.5 * material.λ * logJ^2
end

"""
stress_tensor(material::NeoHookean, F)

Compute first Piola-Kirchhoff stress tensor P for Neo-Hookean material.
P = μ*(F - F⁻ᵀ) + λ*log(J)*F⁻ᵀ
"""
function stress_tensor(material::NeoHookean, F::AbstractMatrix)
    F_inv = inv(F)
    F_invT = F_inv'
    J = det(F)
    logJ = log(J)
    return material.μ * (F - F_invT) + material.λ * logJ * F_invT
end

"""
stress_differential(material::NeoHookean, F)

Compute differential of stress tensor dP/dF for Neo-Hookean material.
Returns (dim²×dim²) matrix representing the tangent stiffness.
"""
function stress_differential(material::NeoHookean, F::AbstractMatrix)
    dim = size(F, 1)
    dim2 = dim * dim
    
    F_inv = inv(F)
    F_invT = F_inv'
    J = det(F)
    logJ = log(J)
    
    # Vectorize F_invT (column-major)
    F_invT_vec = vec(F_invT)
    F_invT_outer = F_invT_vec * F_invT_vec'
    
    # Three components of dP/dF
    D1 = material.μ * Matrix{eltype(F)}(I, dim2, dim2)
    D2 = material.λ * F_invT_outer
    
    # D3 involves tensor contraction - reshape to 4D then permute axes
    coeff = material.λ * logJ - material.μ
    D3_4d = reshape(F_invT_outer, (dim, dim, dim, dim))
    D3 = -coeff * reshape(permutedims(D3_4d, (3, 2, 1, 4)), (dim2, dim2))
    
    return D1 + D2 + D3
end

function get_material_properties(material::NeoHookean)
    return (E = material.E, ν = material.ν, μ = material.μ, λ = material.λ, ρ = material.ρ)
end