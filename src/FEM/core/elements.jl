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

# =============================================================================
# Tetrahedral Elements - Tet4
# =============================================================================

"""
tet4_shape_functions()

Return shape functions and their derivatives for a linear tetrahedral element.
For Tet4, shape functions are constant in physical space, so we compute them
directly from the element geometry.
"""
function tet4_shape_functions(coords::AbstractMatrix{T}) where T
    # Extract vertex coordinates
    x1, y1, z1 = coords[:, 1]
    x2, y2, z2 = coords[:, 2]
    x3, y3, z3 = coords[:, 3]
    x4, y4, z4 = coords[:, 4]
    
    # Volume computation using determinant
    V = abs((x2-x1)*((y3-y1)*(z4-z1) - (y4-y1)*(z3-z1)) -
            (y2-y1)*((x3-x1)*(z4-z1) - (x4-x1)*(z3-z1)) +
            (z2-z1)*((x3-x1)*(y4-y1) - (x4-x1)*(y3-y1))) / 6
    
    # Derivatives of shape functions (constant for linear tets)
    dN_dx = @SVector [
        ((y2-y3)*(z3-z4) - (y3-y4)*(z2-z3)) / (6*V),
        ((y3-y1)*(z1-z4) - (y1-y4)*(z3-z4)) / (6*V),
        ((y1-y2)*(z2-z4) - (y2-y4)*(z1-z2)) / (6*V),
        ((y2-y1)*(z1-z3) - (y1-y3)*(z2-z3)) / (6*V)
    ]
    
    dN_dy = @SVector [
        ((x3-x2)*(z3-z4) - (x3-x4)*(z2-z3)) / (6*V),
        ((x1-x3)*(z1-z4) - (x1-x4)*(z3-z4)) / (6*V),
        ((x2-x1)*(z2-z4) - (x2-x4)*(z1-z2)) / (6*V),
        ((x1-x2)*(z1-z3) - (x1-x3)*(z2-z3)) / (6*V)
    ]
    
    dN_dz = @SVector [
        ((x2-x3)*(y3-y4) - (x3-x4)*(y2-y3)) / (6*V),
        ((x3-x1)*(y1-y4) - (x1-x4)*(y3-y4)) / (6*V),
        ((x1-x2)*(y2-y4) - (x2-x4)*(y1-y2)) / (6*V),
        ((x2-x1)*(y1-y3) - (x1-x3)*(y2-y3)) / (6*V)
    ]
    
    return V, dN_dx, dN_dy, dN_dz
end

"""
tet4_B_matrix(dN_dx, dN_dy, dN_dz)

Construct the 6×12 strain-displacement matrix for a Tet4 element.
"""
function tet4_B_matrix(dN_dx, dN_dy, dN_dz)
    B = zeros(eltype(dN_dx), 6, 12)
    for i in 1:4
        col = 3 * (i - 1) + 1
        B[1, col] = dN_dx[i]        # εxx
        B[2, col + 1] = dN_dy[i]    # εyy
        B[3, col + 2] = dN_dz[i]    # εzz
        B[4, col] = dN_dy[i]        # γxy
        B[4, col + 1] = dN_dx[i]
        B[5, col + 1] = dN_dz[i]    # γyz
        B[5, col + 2] = dN_dy[i]
        B[6, col] = dN_dz[i]        # γxz
        B[6, col + 2] = dN_dx[i]
    end
    return B
end

"""
compute_element_stiffness(mesh, element, material, analysis_type)

Compute the 12×12 stiffness matrix for a Tet4 element.
For linear tetrahedra, the stiffness is computed using single-point integration.
"""
function compute_element_stiffness(
    mesh::Mesh{3,T,:Tet4},
    element::Tet4{T},
    material::LinearElastic{T},
    analysis_type::ThreeDimensional;
) where {T}
    coords = get_element_coords(mesh, element)
    D = get_constitutive_matrix(material, analysis_type)
    
    # Compute shape function derivatives and volume
    V, dN_dx, dN_dy, dN_dz = tet4_shape_functions(coords)
    
    # Construct B matrix
    B = tet4_B_matrix(dN_dx, dN_dy, dN_dz)
    
    # Element stiffness: Ke = V * B' * D * B
    Ke = V * (B' * D * B)
    
    return Ke
end

"""
compute_Dm_inv(coords)

Compute the inverse of the material coordinate matrix Dm for a Tet4 element.
Dm = [X₂-X₁, X₃-X₁, X₄-X₁] where Xᵢ are the reference coordinates.
"""
function compute_Dm_inv(coords::AbstractMatrix{T}) where T
    # Dm matrix: differences from first vertex
    Dm = coords[:, 2:4] .- coords[:, 1:1]
    return inv(Dm)
end

"""
compute_deformation_gradient(coords_current, coords_reference)

Compute the deformation gradient F = Ds * Dm⁻¹ for a Tet4 element.
"""
function compute_deformation_gradient(
    coords_current::AbstractMatrix{T},
    coords_reference::AbstractMatrix{T}
) where T
    Ds = coords_current[:, 2:4] .- coords_current[:, 1:1]
    Dm_inv = compute_Dm_inv(coords_reference)
    return Ds * Dm_inv
end

"""
compute_tet_volume(coords)

Compute the volume of a tetrahedral element.
"""
function compute_tet_volume(coords::AbstractMatrix{T}) where T
    # Volume = |det(Dm)| / 6
    Dm = coords[:, 2:4] .- coords[:, 1:1]
    return abs(det(Dm)) / 6
end

"""
shape_functions(::Type{Tet4}, coords)

Convenience wrapper for Tet4 shape functions.
"""
shape_functions(::Type{Tet4}, coords::AbstractMatrix) = tet4_shape_functions(coords)
