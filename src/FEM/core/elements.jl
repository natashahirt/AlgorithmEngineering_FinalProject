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

    coords = zeros(T, 2, 4)
    for (local_id, node_id) in enumerate(element.nodes)
        coord = mesh.nodes[node_id].coords
        coords[1, local_id] = coord[1]
        coords[2, local_id] = coord[2]
    end

    D = get_constitutive_matrix(material, analysis_type)
    Ke = zeros(promote_type(T, eltype(D)), 8, 8)

    for ξ_raw in GAUSS_POINTS_1D, η_raw in GAUSS_POINTS_1D
        ξ = T(ξ_raw)
        η = T(η_raw)
        _, dN_dξ, dN_dη = quad4_shape_functions(ξ, η)

        dN_nat = hcat(dN_dξ, dN_dη)'
        J = coords * dN_nat'
        detJ = det(J)
        detJ <= zero(detJ) && error("Quad4 element has non-positive Jacobian determinant")

        invJ = inv(J)
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
    coords = zeros(T, 3, 8)
    for (local_id, node_id) in enumerate(element.nodes)
        coord = mesh.nodes[node_id].coords
        coords[1, local_id] = coord[1]
        coords[2, local_id] = coord[2]
        coords[3, local_id] = coord[3]
    end

    D = get_constitutive_matrix(material, analysis_type)
    Ke = zeros(promote_type(T, eltype(D)), 24, 24)

    for ξ_raw in GAUSS_POINTS_1D, η_raw in GAUSS_POINTS_1D, ζ_raw in GAUSS_POINTS_1D
        ξ = T(ξ_raw)
        η = T(η_raw)
        ζ = T(ζ_raw)
        
        _, dN_dξ, dN_dη, dN_dζ = hex8_shape_functions(ξ, η, ζ)

        # Jacobian matrix
        J = zeros(T, 3, 3)
        for i in 1:8
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
        detJ <= zero(detJ) && error("Hex8 element has non-positive Jacobian determinant")

        invJ = inv(J)
        
        # Transform derivatives to physical coordinates
        dN_dx = zeros(T, 8)
        dN_dy = zeros(T, 8)
        dN_dz = zeros(T, 8)
        
        for i in 1:8
            dN_dx[i] = invJ[1,1]*dN_dξ[i] + invJ[1,2]*dN_dη[i] + invJ[1,3]*dN_dζ[i]
            dN_dy[i] = invJ[2,1]*dN_dξ[i] + invJ[2,2]*dN_dη[i] + invJ[2,3]*dN_dζ[i]
            dN_dz[i] = invJ[3,1]*dN_dξ[i] + invJ[3,2]*dN_dη[i] + invJ[3,3]*dN_dζ[i]
        end

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
export quad4_shape_functions, quad4_B_matrix
export compute_element_stiffness, shape_functions
export hex8_shape_functions, hex8_B_matrix
