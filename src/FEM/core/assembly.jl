# assembly.jl - assembly routines
"""
System assembly routines for global stiffness matrices and force vectors.
"""

"""
estimate_nnz(mesh::Mesh)

Estimate the number of non-zero entries in the assembled stiffness matrix
based on element connectivity.
"""
function estimate_nnz(mesh::Mesh{dim,T,EType}) where {dim,T,EType}
    # Get a representative element to determine DOFs per element
    elem = first(mesh.elements)
    elem_dofs = get_element_dofs(mesh, elem)
    dofs_per_elem = length(elem_dofs)
    
    # Each element contributes dofs_per_elem^2 entries – favour an over-estimate
    # to avoid having to grow the triplet buffers during assembly.
    return max(length(mesh.elements) * dofs_per_elem^2, 1)
end

"""
assemble_stiffness_matrix(mesh, material, analysis_type; kwargs...)

Assemble the global stiffness matrix for a mesh. Generic implementation that
works for all element types through multiple dispatch on compute_element_stiffness.

# Performance optimization: Preallocates triplet arrays based on mesh structure.
"""
function assemble_stiffness_matrix(
    mesh::Mesh{dim,T,EType},
    material::LinearElastic{T},
    analysis_type;
    kwargs...
) where {dim,T,EType}
    
    ndof = maximum(last(node.dofs) for node in mesh.nodes)
    
    # Preallocate triplet arrays based on mesh structure
    nnz_estimate = estimate_nnz(mesh)
    row_indices = Vector{Int}(undef, nnz_estimate)
    col_indices = Vector{Int}(undef, nnz_estimate)
    values = Vector{T}(undef, nnz_estimate)
    
    idx = 1
    for element in mesh.elements
        # Compute element stiffness (dispatches based on element type)
        Ke = compute_element_stiffness(mesh, element, material, analysis_type; kwargs...)
        elem_dofs = get_element_dofs(mesh, element)
        
        # Add element contributions to triplet arrays
        for (local_i, dof_i) in enumerate(elem_dofs)
            for (local_j, dof_j) in enumerate(elem_dofs)
                if idx > length(row_indices)
                    current_len = length(row_indices)
                    growth = max(current_len >>> 1, 1)  # 50% growth, minimum 1
                    new_len = max(idx, current_len + growth)
                    resize!(row_indices, new_len)
                    resize!(col_indices, new_len)
                    resize!(values, new_len)
                end
                row_indices[idx] = dof_i
                col_indices[idx] = dof_j
                values[idx] = Ke[local_i, local_j]
                idx += 1
            end
        end
    end
    
    # Trim arrays to actual size (estimate may be slightly off)
    resize!(row_indices, idx - 1)
    resize!(col_indices, idx - 1)
    resize!(values, idx - 1)
    
    return sparse(row_indices, col_indices, values, ndof, ndof)
end

"""
assemble_force_vector(loads, ndof)

Assemble a global force vector from a dictionary that maps global DOF indices
to force magnitudes. Unified implementation with automatic type inference.
"""
function assemble_force_vector(loads::Dict, ndof::Int)
    # Infer type from first value in dict, default to Float64 if empty
    T = isempty(loads) ? Float64 : promote_type(typeof(first(values(loads))), Float64)
    F = zeros(T, ndof)
    for (dof, force) in loads
        1 <= dof <= ndof || continue
        F[dof] += convert(T, force)
    end
    return F
end

"""
apply_boundary_conditions(K, F, constrained_dofs, prescribed_values)

Apply Dirichlet boundary conditions by zeroing-out the constrained rows and
columns while honouring non-zero prescribed displacement values.

Unified implementation that works efficiently for both sparse and dense matrices.
"""
function apply_boundary_conditions(
    K::AbstractMatrix{T},
    F,
    constrained_dofs::AbstractVector{<:Integer},
    prescribed_values::AbstractVector = zeros(eltype(F), length(constrained_dofs)),
) where {T}

    length(constrained_dofs) == length(prescribed_values) ||
        error("constrained_dofs and prescribed_values must have the same length")

    K_mod = copy(K)
    F_mod = copy(F)

    # Optimize for sparse matrices
    if K isa SparseMatrixCSC
        for (dof, value) in zip(constrained_dofs, prescribed_values)
            1 <= dof <= size(K_mod, 1) || continue

            # Extract column and adjust force vector
            col = Array(K_mod[:, dof])
            F_mod .-= col .* value

            # Zero out row and column in sparse structure
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
    else
        # Dense matrix path (rarely used)
        for (dof, value) in zip(constrained_dofs, prescribed_values)
            1 <= dof <= size(K_mod, 1) || continue
            F_mod .-= K_mod[:, dof] .* value
            K_mod[dof, :] .= 0
            K_mod[:, dof] .= 0
            K_mod[dof, dof] = one(eltype(K_mod))
            F_mod[dof] = value
        end
    end

    return K_mod, F_mod
end
