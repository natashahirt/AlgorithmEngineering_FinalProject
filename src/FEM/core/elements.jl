# elements.jl - element computations shared by 2D and 3D analyses
"""
Element-level computations for supported finite elements.
"""

const GAUSS_POINTS_1D = (-1/√3, 1/√3)

"""
quad4_shape_functions(ξ, η)

Return shape functions and derivatives for Quad4 element.
"""
function quad4_shape_functions(ξ, η)
    N = @SVector [0.25 * (1 - ξ) * (1 - η),
                  0.25 * (1 + ξ) * (1 - η),
                  0.25 * (1 + ξ) * (1 + η),
                  0.25 * (1 - ξ) * (1 + η)]

    dN_dξ = @SVector [-0.25 * (1 - η),
                       0.25 * (1 - η),
                       0.25 * (1 + η),
                      -0.25 * (1 + η)]

    dN_dη = @SVector [-0.25 * (1 - ξ),
                      -0.25 * (1 + ξ),
                       0.25 * (1 + ξ),
                       0.25 * (1 - ξ)]

    return N, dN_dξ, dN_dη
end

"""
compute_jacobian_2d(coords, dN_dξ, dN_dη)

Compute Jacobian matrix, determinant, and inverse for 2D element.
Returns (J, detJ, invJ).
"""
function compute_jacobian_2d(coords, dN_dξ, dN_dη)
    dN_nat = hcat(dN_dξ, dN_dη)'
    J = coords * dN_nat'
    detJ = det(J)
    detJ <= zero(detJ) && error("Element has non-positive Jacobian determinant")
    return J, detJ, inv(J)
end

"""
quad4_B_matrix(dN_dx, dN_dy)

Construct 3×8 strain-displacement matrix for Quad4 element.
"""
function quad4_B_matrix(dN_dx, dN_dy)
    B = zeros(eltype(dN_dx), 3, 8)
    for i in 1:4
        col = 2 * (i - 1) + 1
        B[1, col] = dN_dx[i]
        B[2, col + 1] = dN_dy[i]
        B[3, col] = dN_dy[i]
        B[3, col + 1] = dN_dx[i]
    end
    return B
end

"""
compute_B_matrix(mesh, element, analysis_type, ξ=0, η=0, ζ=0)

Compute strain-displacement matrix B at specified natural coordinates.
Default evaluation is at element center (ξ=η=ζ=0).
"""
function compute_B_matrix(
    mesh::Mesh{2,T,:Quad4},
    element::Quad4{T},
    ::Union{PlaneStress, PlaneStrain},
    ξ::T = zero(T),
    η::T = zero(T)
) where {T}
    coords = get_element_coords(mesh, element)
    
    _, dN_dξ, dN_dη = quad4_shape_functions(ξ, η)
    J, detJ, invJ = compute_jacobian_2d(coords, dN_dξ, dN_dη)
    dN_nat = hcat(dN_dξ, dN_dη)'
    dN_cart = invJ * dN_nat
    dN_dx = view(dN_cart, 1, :)
    dN_dy = view(dN_cart, 2, :)
    
    return quad4_B_matrix(dN_dx, dN_dy)
end

"""
compute_element_strain!(ϵ_buffer, mesh, element, u_elem, analysis_type)

Compute element strain vector: ε = B * u_elem
Evaluates at element centroid (ξ=0, η=0) for Quad4.
Dispatches on analysis type (PlaneStress or PlaneStrain).
"""
function compute_element_strain!(
    ϵ_buffer::AbstractMatrix{T},
    mesh::Mesh{2,T,:Quad4},
    element::Quad4{T},
    u_elem::AbstractVector{T},
    analysis_type::Union{PlaneStress, PlaneStrain}
) where {T}
    # Compute B matrix at element center
    B = compute_B_matrix(mesh, element, analysis_type)
    
    # Compute strain: ε = B * u
    mul!(ϵ_buffer, B, u_elem)
    
    return nothing
end

"""
compute_element_stiffness(mesh, element, material, analysis_type; thickness=1)

Compute 8×8 stiffness matrix for Quad4 element using 2×2 Gauss quadrature.
"""
function compute_element_stiffness(
    mesh::Mesh{2,T,:Quad4},
    element::Quad4{T},
    material::LinearElastic{T},
    analysis_type::Union{PlaneStress,PlaneStrain};
    thickness::T = one(T),
) where {T}

    coords = get_element_coords(mesh, element)
    D = get_constitutive_matrix(material, analysis_type)
    Ke = zeros(promote_type(T, eltype(D)), 8, 8)

    for ξ_raw in GAUSS_POINTS_1D, η_raw in GAUSS_POINTS_1D
        ξ = T(ξ_raw)
        η = T(η_raw)
        _, dN_dξ, dN_dη = quad4_shape_functions(ξ, η)

        J, detJ, invJ = compute_jacobian_2d(coords, dN_dξ, dN_dη)
        dN_nat = hcat(dN_dξ, dN_dη)'
        dN_cart = invJ * dN_nat
        dN_dx = view(dN_cart, 1, :)
        dN_dy = view(dN_cart, 2, :)

        B = quad4_B_matrix(dN_dx, dN_dy)
        Ke .+= (B' * D * B) * detJ * thickness
    end

    return Ke
end

"""
shape_functions(::Type{Quad4}, ξ, η)

Convenience wrapper for Quad4 shape functions.
"""
shape_functions(::Type{Quad4}, ξ, η) = quad4_shape_functions(ξ, η)

# =============================================================================
# 3D Elements - Hexahedral (Hex8)
# =============================================================================

"""
hex8_shape_functions(ξ, η, ζ)

Return shape functions and their derivatives in natural coordinates for a
8-node hexahedral element.
"""
function hex8_shape_functions(ξ, η, ζ)
    N = @SVector [
        0.125 * (1 - ξ) * (1 - η) * (1 - ζ),
        0.125 * (1 + ξ) * (1 - η) * (1 - ζ),
        0.125 * (1 + ξ) * (1 + η) * (1 - ζ),
        0.125 * (1 - ξ) * (1 + η) * (1 - ζ),
        0.125 * (1 - ξ) * (1 - η) * (1 + ζ),
        0.125 * (1 + ξ) * (1 - η) * (1 + ζ),
        0.125 * (1 + ξ) * (1 + η) * (1 + ζ),
        0.125 * (1 - ξ) * (1 + η) * (1 + ζ)
    ]

    dN_dξ = @SVector [
        -0.125 * (1 - η) * (1 - ζ),
         0.125 * (1 - η) * (1 - ζ),
         0.125 * (1 + η) * (1 - ζ),
        -0.125 * (1 + η) * (1 - ζ),
        -0.125 * (1 - η) * (1 + ζ),
         0.125 * (1 - η) * (1 + ζ),
         0.125 * (1 + η) * (1 + ζ),
        -0.125 * (1 + η) * (1 + ζ)
    ]

    dN_dη = @SVector [
        -0.125 * (1 - ξ) * (1 - ζ),
        -0.125 * (1 + ξ) * (1 - ζ),
         0.125 * (1 + ξ) * (1 - ζ),
         0.125 * (1 - ξ) * (1 - ζ),
        -0.125 * (1 - ξ) * (1 + ζ),
        -0.125 * (1 + ξ) * (1 + ζ),
         0.125 * (1 + ξ) * (1 + ζ),
         0.125 * (1 - ξ) * (1 + ζ)
    ]

    dN_dζ = @SVector [
        -0.125 * (1 - ξ) * (1 - η),
        -0.125 * (1 + ξ) * (1 - η),
        -0.125 * (1 + ξ) * (1 + η),
        -0.125 * (1 - ξ) * (1 + η),
         0.125 * (1 - ξ) * (1 - η),
         0.125 * (1 + ξ) * (1 - η),
         0.125 * (1 + ξ) * (1 + η),
         0.125 * (1 - ξ) * (1 + η)
    ]

    return N, dN_dξ, dN_dη, dN_dζ
end

"""
compute_jacobian_3d(coords, dN_dξ, dN_dη, dN_dζ)

Compute Jacobian matrix, determinant, and inverse for 3D element.
Returns (J, detJ, invJ).
"""
function compute_jacobian_3d(coords, dN_dξ, dN_dη, dN_dζ)
    T = eltype(coords)
    J = zeros(T, 3, 3)
    n_nodes = size(coords, 2)
    
    for i in 1:n_nodes
        J[1, 1] += coords[1, i] * dN_dξ[i]
        J[1, 2] += coords[1, i] * dN_dη[i]
        J[1, 3] += coords[1, i] * dN_dζ[i]
        J[2, 1] += coords[2, i] * dN_dξ[i]
        J[2, 2] += coords[2, i] * dN_dη[i]
        J[2, 3] += coords[2, i] * dN_dζ[i]
        J[3, 1] += coords[3, i] * dN_dξ[i]
        J[3, 2] += coords[3, i] * dN_dη[i]
        J[3, 3] += coords[3, i] * dN_dζ[i]
    end
    
    detJ = det(J)
    detJ <= zero(detJ) && error("Element has non-positive Jacobian determinant")
    return J, detJ, inv(J)
end

"""
hex8_B_matrix(dN_dx, dN_dy, dN_dz)

Construct the 6×24 strain-displacement matrix for a Hex8 element.
"""
function hex8_B_matrix(dN_dx, dN_dy, dN_dz)
    B = zeros(eltype(dN_dx), 6, 24)
    for i in 1:8
        col = 3 * (i - 1) + 1
        B[1, col] = dN_dx[i]      # εxx
        B[2, col + 1] = dN_dy[i]  # εyy
        B[3, col + 2] = dN_dz[i]  # εzz
        B[4, col] = dN_dy[i]      # γxy
        B[4, col + 1] = dN_dx[i]
        B[5, col + 1] = dN_dz[i]  # γyz
        B[5, col + 2] = dN_dy[i]
        B[6, col] = dN_dz[i]      # γxz
        B[6, col + 2] = dN_dx[i]
    end
    return B
end

"""
compute_B_matrix(mesh, element, analysis_type, ξ=0, η=0, ζ=0)

Compute strain-displacement matrix B at specified natural coordinates for 3D.
Default evaluation is at element center (ξ=η=ζ=0).
"""
function compute_B_matrix(
    mesh::Mesh{3,T,:Hex8},
    element::Hex8{T},
    ::ThreeDimensional,
    ξ::T = zero(T),
    η::T = zero(T),
    ζ::T = zero(T)
) where {T}
    coords = get_element_coords(mesh, element)
    
    _, dN_dξ, dN_dη, dN_dζ = hex8_shape_functions(ξ, η, ζ)
    J, detJ, invJ = compute_jacobian_3d(coords, dN_dξ, dN_dη, dN_dζ)
    dN_nat = hcat(dN_dξ, dN_dη, dN_dζ)
    dN_cart = (invJ * dN_nat')'
    dN_dx = @view dN_cart[:, 1]
    dN_dy = @view dN_cart[:, 2]
    dN_dz = @view dN_cart[:, 3]
    
    return hex8_B_matrix(dN_dx, dN_dy, dN_dz)
end

"""
compute_element_strain!(ϵ_buffer, mesh, element, u_elem, analysis_type)

Compute element strain vector: ε = B * u_elem
Evaluates at element centroid (ξ=η=ζ=0) for Hex8.
Dispatches on analysis type (ThreeDimensional).
"""
function compute_element_strain!(
    ϵ_buffer::AbstractMatrix{T},
    mesh::Mesh{3,T,:Hex8},
    element::Hex8{T},
    u_elem::AbstractVector{T},
    analysis_type::ThreeDimensional
) where {T}
    # Compute B matrix at element center
    B = compute_B_matrix(mesh, element, analysis_type)
    
    # Compute strain: ε = B * u
    mul!(ϵ_buffer, B, u_elem)
    
    return nothing
end

"""
compute_element_stiffness(mesh, element, material, analysis_type)

Compute the 24×24 stiffness matrix for a Hex8 element using 2×2×2 Gauss
quadrature for 3D analysis.
"""
function compute_element_stiffness(
    mesh::Mesh{3,T,:Hex8},
    element::Hex8{T},
    material::LinearElastic{T},
    analysis_type::ThreeDimensional;
) where {T}
    coords = get_element_coords(mesh, element)
    D = get_constitutive_matrix(material, analysis_type)
    Ke = zeros(promote_type(T, eltype(D)), 24, 24)

    for ξ_raw in GAUSS_POINTS_1D, η_raw in GAUSS_POINTS_1D, ζ_raw in GAUSS_POINTS_1D
        ξ = T(ξ_raw)
        η = T(η_raw)
        ζ = T(ζ_raw)
        
        _, dN_dξ, dN_dη, dN_dζ = hex8_shape_functions(ξ, η, ζ)

        J, detJ, invJ = compute_jacobian_3d(coords, dN_dξ, dN_dη, dN_dζ)
        
        # Transform derivatives to physical coordinates using matrix multiplication
        dN_nat = hcat(dN_dξ, dN_dη, dN_dζ)  # 8×3
        dN_cart = (invJ * dN_nat')'          # (3×3) * (3×8) = (3×8), then transpose to 8×3
        dN_dx = @view dN_cart[:, 1]
        dN_dy = @view dN_cart[:, 2]
        dN_dz = @view dN_cart[:, 3]

        B = hex8_B_matrix(dN_dx, dN_dy, dN_dz)
        Ke .+= (B' * D * B) * detJ
    end

    return Ke
end

"""
shape_functions(::Type{Hex8}, ξ, η, ζ)

Convenience wrapper for Hex8 shape functions.
"""
shape_functions(::Type{Hex8}, ξ, η, ζ) = hex8_shape_functions(ξ, η, ζ)
