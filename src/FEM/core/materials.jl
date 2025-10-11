# materials.jl - material properties
"""
Material properties and constitutive models.
"""

export Material, LinearElastic, PlaneStress, PlaneStrain, ThreeDimensional
export get_constitutive_matrix, get_material_properties

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
PlaneStress

Analysis type for plane stress conditions.
"""
struct PlaneStress end

"""
PlaneStrain

Analysis type for plane strain conditions.
"""
struct PlaneStrain end

"""
ThreeDimensional

Analysis type for 3D conditions.
"""
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

# Placeholder for advanced material models
"""
NonlinearMaterial

Abstract type for nonlinear material models.
"""
abstract type NonlinearMaterial <: Material end

"""
Plasticity

Plasticity material model.
"""
struct Plasticity <: NonlinearMaterial
    # TODO: Implement plasticity model
    # - Yield criteria (von Mises, Tresca, etc.)
    # - Hardening rules (isotropic, kinematic)
    # - Flow rules
end

"""
Hyperelastic

Hyperelastic material model.
"""
struct Hyperelastic <: NonlinearMaterial
    # TODO: Implement hyperelastic models
    # - Neo-Hookean
    # - Mooney-Rivlin
    # - Ogden
end