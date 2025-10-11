# assembly.jl - assembly routines
"""
System assembly routines for global stiffness matrices and force vectors.
"""

"""
assemble_stiffness_matrix(mesh, material, analysis_type; thickness=1)

Assemble the global stiffness matrix for a mesh. Dispatch selects the correct
integration routine for the mesh dimensionality.
"""
function assemble_stiffness_matrix(
    mesh::Mesh{2,T,:Quad4},
    material::LinearElastic{T},
    analysis_type::Union{PlaneStress,PlaneStrain};
    thickness::T = one(T),
) where {T}

    ndof = maximum(last(node.dofs) for node in mesh.nodes)

    row_indices = Int[]
    col_indices = Int[]
    values = T[]

    for element in mesh.elements
        Ke = compute_element_stiffness(mesh, element, material, analysis_type; thickness=thickness)
        elem_dofs = get_element_dofs(mesh, element)

        for (local_i, dof_i) in enumerate(elem_dofs)
            for (local_j, dof_j) in enumerate(elem_dofs)
                push!(row_indices, dof_i)
                push!(col_indices, dof_j)
                push!(values, T(Ke[local_i, local_j]))
            end
        end
    end

    return sparse(row_indices, col_indices, values, ndof, ndof)
end

function assemble_stiffness_matrix(
    mesh::Mesh{3,T,:Hex8},
    material::LinearElastic{T},
    analysis_type::ThreeDimensional;
) where {T}

    ndof = maximum(last(node.dofs) for node in mesh.nodes)

    row_indices = Int[]
    col_indices = Int[]
    values = T[]

    for element in mesh.elements
        Ke = compute_element_stiffness(mesh, element, material, analysis_type)
        elem_dofs = get_element_dofs(mesh, element)

        for (local_i, dof_i) in enumerate(elem_dofs)
            for (local_j, dof_j) in enumerate(elem_dofs)
                push!(row_indices, dof_i)
                push!(col_indices, dof_j)
                push!(values, T(Ke[local_i, local_j]))
            end
        end
    end

    return sparse(row_indices, col_indices, values, ndof, ndof)
end

"""
assemble_force_vector(loads, ndof)

Assemble a global force vector from a dictionary that maps global DOF indices
to force magnitudes.
"""
function assemble_force_vector(loads::Dict{Int,T}, ndof::Int) where {T<:Real}
    F = zeros(T, ndof)
    for (dof, force) in loads
        1 <= dof <= ndof || continue
        F[dof] += force
    end
    return F
end

function assemble_force_vector(loads::Dict, ndof::Int)
    F = zeros(Float64, ndof)
    for (dof, force) in loads
        1 <= dof <= ndof || continue
        F[dof] += Float64(force)
    end
    return F
end

"""
apply_boundary_conditions(K, F, constrained_dofs, prescribed_values)

Apply Dirichlet boundary conditions by zeroing-out the constrained rows and
columns while honouring non-zero prescribed displacement values.
"""
function apply_boundary_conditions(
    K::SparseMatrixCSC{T},
    F,
    constrained_dofs::AbstractVector{<:Integer},
    prescribed_values::AbstractVector = zeros(eltype(F), length(constrained_dofs)),
) where {T}

    length(constrained_dofs) == length(prescribed_values) ||
        error("constrained_dofs and prescribed_values must have the same length")

    K_mod = copy(K)
    F_mod = copy(F)

    for (dof, value) in zip(constrained_dofs, prescribed_values)
        1 <= dof <= size(K_mod, 1) || continue

        col = Array(K_mod[:, dof])
        F_mod .-= col .* value

        for col_idx in 1:size(K_mod, 2)
            col_start = K_mod.colptr[col_idx]
            col_end = K_mod.colptr[col_idx + 1] - 1
            for idx in col_start:col_end
                if K_mod.rowval[idx] == dof || col_idx == dof
                    K_mod.nzval[idx] = zero(T)
                end
            end
        end

        K_mod[dof, dof] = one(T)
        F_mod[dof] = value
    end

    dropzeros!(K_mod)
    return K_mod, F_mod
end

function apply_boundary_conditions(
    K::AbstractMatrix,
    F,
    constrained_dofs::AbstractVector{<:Integer},
    prescribed_values::AbstractVector = zeros(eltype(F), length(constrained_dofs)),
)
    length(constrained_dofs) == length(prescribed_values) ||
        error("constrained_dofs and prescribed_values must have the same length")

    K_mod = copy(K)
    F_mod = copy(F)

    for (dof, value) in zip(constrained_dofs, prescribed_values)
        1 <= dof <= size(K_mod, 1) || continue
        F_mod .-= K_mod[:, dof] .* value
        K_mod[dof, :] .= 0
        K_mod[:, dof] .= 0
        K_mod[dof, dof] = one(eltype(K_mod))
        F_mod[dof] = value
    end

    return K_mod, F_mod
end

export assemble_stiffness_matrix, assemble_force_vector, apply_boundary_conditions
