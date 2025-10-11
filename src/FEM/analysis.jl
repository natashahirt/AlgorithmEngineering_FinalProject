# analysis.jl - displacement and stress computation
"""
Displacement and stress computation utilities that work for both 2D and 3D
analyses.
"""

export solve_displacements, compute_element_stresses, compute_nodal_stresses
export compute_strain_energy, compute_compliance, extract_displacements
export compute_von_mises_stress, compute_principal_stresses

"""
solve_displacements(K, f)

Solve the linear system `K * u = f` for the displacement vector.
"""
function solve_displacements(K, f)
    try
        return K \ f
    catch e
        @warn "Direct solver failed: $e"
        return pinv(K) * f
    end
end

# -----------------------------------------------------------------------------
# Element-level stress evaluation
# -----------------------------------------------------------------------------

"""
compute_element_stresses(mesh, material, displacements, element_id, analysis_type)

Compute the Cauchy stress vector at the centre of the requested element.
For 2D elements the returned vector is `[σxx, σyy, σxy]`. For 3D elements it is
`[σxx, σyy, σzz, σxy, σxz, σyz]`.
"""
function compute_element_stresses(
    mesh::Mesh{2,T,:Quad4},
    material::LinearElastic{T},
    displacements::AbstractVector,
    element_id::Int,
    analysis_type::Union{PlaneStress,PlaneStrain};
    ξ::Real = 0,
    η::Real = 0,
) where {T}

    element = mesh.elements[element_id]

    coords = zeros(T, 2, 4)
    for (local_id, node_id) in enumerate(element.nodes)
        coord = mesh.nodes[node_id].coords
        coords[1, local_id] = coord[1]
        coords[2, local_id] = coord[2]
    end

    _, dN_dξ, dN_dη = quad4_shape_functions(T(ξ), T(η))
    dN_nat = hcat(dN_dξ, dN_dη)'
    J = coords * dN_nat'
    detJ = det(J)
    detJ <= zero(detJ) && error("Quad4 element has non-positive Jacobian determinant")

    invJ = inv(J)
    dN_cart = invJ * dN_nat
    dN_dx = view(dN_cart, 1, :)
    dN_dy = view(dN_cart, 2, :)

    B = quad4_B_matrix(dN_dx, dN_dy)
    elem_dofs = get_element_dofs(mesh, element)
    u_e = displacements[elem_dofs]

    strain = B * u_e
    D = get_constitutive_matrix(material, analysis_type)
    stress = D * strain

    return stress
end

function compute_element_stresses(
    mesh::Mesh{3,T,:Hex8},
    material::LinearElastic{T},
    displacements::AbstractVector,
    element_id::Int,
    analysis_type::ThreeDimensional;
    ξ::Real = 0,
    η::Real = 0,
    ζ::Real = 0,
) where {T}

    element = mesh.elements[element_id]

    coords = zeros(T, 3, 8)
    for (local_id, node_id) in enumerate(element.nodes)
        coord = mesh.nodes[node_id].coords
        coords[1, local_id] = coord[1]
        coords[2, local_id] = coord[2]
        coords[3, local_id] = coord[3]
    end

    _, dN_dξ, dN_dη, dN_dζ = hex8_shape_functions(T(ξ), T(η), T(ζ))

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
    dN_dx = zeros(T, 8)
    dN_dy = zeros(T, 8)
    dN_dz = zeros(T, 8)

    for i in 1:8
        dN_dx[i] = invJ[1, 1] * dN_dξ[i] + invJ[1, 2] * dN_dη[i] + invJ[1, 3] * dN_dζ[i]
        dN_dy[i] = invJ[2, 1] * dN_dξ[i] + invJ[2, 2] * dN_dη[i] + invJ[2, 3] * dN_dζ[i]
        dN_dz[i] = invJ[3, 1] * dN_dξ[i] + invJ[3, 2] * dN_dη[i] + invJ[3, 3] * dN_dζ[i]
    end

    B = hex8_B_matrix(dN_dx, dN_dy, dN_dz)
    elem_dofs = get_element_dofs(mesh, element)
    u_e = displacements[elem_dofs]

    strain = B * u_e
    D = get_constitutive_matrix(material, analysis_type)
    stress = D * strain

    return stress
end

# -----------------------------------------------------------------------------
# Nodal stress recovery
# -----------------------------------------------------------------------------

"""
compute_nodal_stresses(mesh, material, displacements, analysis_type)

Recover nodal stresses by averaging the element stresses that touch each node.
"""
function compute_nodal_stresses(
    mesh::Mesh{2,T,:Quad4},
    material::LinearElastic{T},
    displacements::AbstractVector,
    analysis_type::Union{PlaneStress,PlaneStrain},
) where {T}

    num_nodes = length(mesh.nodes)
    stresses = zeros(Float64, num_nodes, 3)
    counts = zeros(Int, num_nodes)

    for (elem_id, element) in enumerate(mesh.elements)
        σ = compute_element_stresses(mesh, material, displacements, elem_id, analysis_type)
        for node_id in element.nodes
            for i in 1:3
                stresses[node_id, i] += Float64(σ[i])
            end
            counts[node_id] += 1
        end
    end

    for node_id in 1:num_nodes
        if counts[node_id] > 0
            stresses[node_id, :] ./= counts[node_id]
        end
    end

    return stresses
end

function compute_nodal_stresses(
    mesh::Mesh{3,T,:Hex8},
    material::LinearElastic{T},
    displacements::AbstractVector,
    analysis_type::ThreeDimensional,
) where {T}

    num_nodes = length(mesh.nodes)
    stresses = zeros(Float64, num_nodes, 6)
    counts = zeros(Int, num_nodes)

    for (elem_id, element) in enumerate(mesh.elements)
        σ = compute_element_stresses(mesh, material, displacements, elem_id, analysis_type)
        for node_id in element.nodes
            for i in 1:6
                stresses[node_id, i] += Float64(σ[i])
            end
            counts[node_id] += 1
        end
    end

    for node_id in 1:num_nodes
        if counts[node_id] > 0
            stresses[node_id, :] ./= counts[node_id]
        end
    end

    return stresses
end

# -----------------------------------------------------------------------------
# Energy and compliance
# -----------------------------------------------------------------------------

"""
compute_strain_energy(K, displacements)

Compute the elastic strain energy `0.5 * u' * K * u`.
"""
compute_strain_energy(K, displacements) = 0.5 * dot(displacements, K * displacements)

"""
compute_compliance(f, displacements)

Compute the structural compliance `f' * u`.
"""
compute_compliance(f, displacements) = dot(f, displacements)

# -----------------------------------------------------------------------------
# Displacement extraction
# -----------------------------------------------------------------------------

"""
extract_displacements(mesh, u_global, node_ids)

Return the displacement components for the specified nodes as a matrix where
each row corresponds to a node and columns correspond to local directions.
"""
function extract_displacements(mesh::Mesh, u_global::AbstractVector, node_ids::Vector{Int})
    dofs_per_node = get_dofs_per_node(mesh)
    result = zeros(eltype(u_global), length(node_ids), dofs_per_node)

    for (row, node_id) in enumerate(node_ids)
        for direction in 1:dofs_per_node
            dof = compute_global_dof(node_id, direction, mesh)
            if 1 <= dof <= length(u_global)
                result[row, direction] = u_global[dof]
            else
                result[row, direction] = zero(eltype(u_global))
            end
        end
    end

    return result
end

"""
extract_displacements(mesh, u_global, node_ids, direction)

Return the displacement in a single direction for each requested node.
"""
function extract_displacements(
    mesh::Mesh,
    u_global::AbstractVector,
    node_ids::Vector{Int},
    direction::Int,
)
    dofs_per_node = get_dofs_per_node(mesh)
    1 <= direction <= dofs_per_node ||
        error("Direction $direction not valid for $(dofs_per_node)D mesh")

    result = zeros(eltype(u_global), length(node_ids))
    for (idx, node_id) in enumerate(node_ids)
        dof = compute_global_dof(node_id, direction, mesh)
        if 1 <= dof <= length(u_global)
            result[idx] = u_global[dof]
        else
            result[idx] = zero(eltype(u_global))
        end
    end

    return result
end

# -----------------------------------------------------------------------------
# Stress post-processing
# -----------------------------------------------------------------------------

"""
compute_von_mises_stress(stresses)

Compute the von Mises equivalent stress from the supplied stress components.
"""
function compute_von_mises_stress(stresses)
    if length(stresses) == 3
        σxx, σyy, σxy = stresses
        return sqrt(σxx^2 - σxx * σyy + σyy^2 + 3 * σxy^2)
    elseif length(stresses) == 6
        σxx, σyy, σzz, σxy, σxz, σyz = stresses
        return sqrt(
            0.5 * (
                (σxx - σyy)^2 +
                (σyy - σzz)^2 +
                (σzz - σxx)^2 +
                6 * (σxy^2 + σxz^2 + σyz^2)
            ),
        )
    else
        error("Invalid stress vector length: $(length(stresses))")
    end
end

"""
compute_principal_stresses(stresses)

Compute the principal stresses from the supplied stress components.
"""
function compute_principal_stresses(stresses)
    if length(stresses) == 3
        σxx, σyy, σxy = stresses
        σ_avg = (σxx + σyy) / 2
        τ_max = sqrt(((σxx - σyy) / 2)^2 + σxy^2)
        σ1 = σ_avg + τ_max
        σ2 = σ_avg - τ_max
        return (σ1, σ2, 0.0)
    elseif length(stresses) == 6
        σxx, σyy, σzz, σxy, σxz, σyz = stresses
        σ = [
            σxx  σxy  σxz
            σxy  σyy  σyz
            σxz  σyz  σzz
        ]
        eigenvals = eigvals(σ)
        σ1, σ2, σ3 = sort(eigenvals, rev=true)
        return (σ1, σ2, σ3)
    else
        error("Invalid stress vector length: $(length(stresses))")
    end
end
