# solvers.jl - FEM solvers for linear and nonlinear problems
"""
Solvers for finite element analysis, including linear direct solvers and
nonlinear Newton-Raphson methods with line search.
"""

# =============================================================================
# Linear Solvers
# =============================================================================

"""
solve_displacements(K, f)

Solve the linear system `K * u = f` for the displacement vector.
Uses direct solver with fallback to pseudo-inverse if singular.
"""
function solve_displacements(K, f)
    try
        return K \ f
    catch e
        @warn "Direct solver failed: $e"
        return pinv(K) * f
    end
end

# =============================================================================
# Nonlinear Analysis - Elastic Forces
# =============================================================================

"""
compute_elastic_forces(mesh, material, displacements)

Compute internal elastic forces for nonlinear materials using deformation gradient.
Only implemented for Tet4 elements with NonlinearMaterial models.

Returns a force vector with same size as displacement vector.
"""
function compute_elastic_forces(
    mesh::Mesh{3,T,:Tet4},
    material::NonlinearMaterial,
    displacements::AbstractVector{T},
) where T
    num_nodes = length(mesh.nodes)
    forces = zeros(T, num_nodes * 3)
    
    for (elem_id, element) in enumerate(mesh.elements)
        # Get reference coordinates
        coords_ref = get_element_coords(mesh, element)
        
        # Get current coordinates (reference + displacement)
        coords_curr = copy(coords_ref)
        for (local_id, node_id) in enumerate(element.nodes)
            for dim in 1:3
                dof = compute_global_dof(node_id, dim, mesh)
                coords_curr[dim, local_id] += displacements[dof]
            end
        end
        
        # Compute deformation gradient F
        F = compute_deformation_gradient(coords_curr, coords_ref)
        
        # Compute first Piola-Kirchhoff stress tensor P
        P = stress_tensor(material, F)
        
        # Compute element volume
        V = compute_tet_volume(coords_ref)
        
        # Compute Dm_inv for this element
        Dm_inv = compute_Dm_inv(coords_ref)
        
        # Compute dF/dx (simplified for tets)
        # For a tet, we need the gradient of each entry of F w.r.t. nodal positions
        dF_dx = compute_dF_dx_tet4(Dm_inv)
        
        # Compute nodal forces: f = -V * dF/dx' * vec(P)
        P_vec = vec(P')  # Vectorize P (column-major)
        dE_dx = V * dF_dx' * P_vec
        
        # Assemble into global force vector
        elem_dofs = get_element_dofs(mesh, element)
        forces[elem_dofs] .-= dE_dx
    end
    
    return forces
end

"""
compute_dF_dx_tet4(Dm_inv)

Compute the derivative matrix dF/dx for a tetrahedral element.
Returns a (9×12) matrix relating deformation gradient to nodal displacements.
"""
function compute_dF_dx_tet4(Dm_inv::AbstractMatrix{T}) where T
    dim = 3
    dF_dx = zeros(T, 9, 12)
    
    # dF/dx1 (node 1)
    for i in 1:dim
        for j in 1:dim
            F_idx = (j-1)*dim + i  # Index into vectorized F
            dF_dx[F_idx, i] = -sum(Dm_inv[j, :])
        end
    end
    
    # dF/dx2, dx3, dx4 (nodes 2, 3, 4)
    for node in 1:3  # Nodes 2, 3, 4
        for i in 1:dim
            for j in 1:dim
                F_idx = (j-1)*dim + i
                x_idx = dim*node + i
                dF_dx[F_idx, x_idx] = Dm_inv[j, node]
            end
        end
    end
    
    return dF_dx
end

"""
assemble_nonlinear_tangent(mesh, material, displacements)

Assemble the consistent tangent stiffness matrix for Tet4 elements with nonlinear
materials by linearising the first Piola-Kirchhoff stress about the current state.
"""
function assemble_nonlinear_tangent(
    mesh::Mesh{3,T,:Tet4},
    material::NonlinearMaterial,
    displacements::AbstractVector{T},
) where T
    ndof = maximum(last(node.dofs) for node in mesh.nodes)
    nnz_estimate = max(length(mesh.elements) * 12^2, 1)
    row_indices = Vector{Int}(undef, nnz_estimate)
    col_indices = Vector{Int}(undef, nnz_estimate)
    values = Vector{T}(undef, nnz_estimate)

    idx = 1
    for element in mesh.elements
        coords_ref = get_element_coords(mesh, element)
        coords_curr = copy(coords_ref)
        for (local_id, node_id) in enumerate(element.nodes)
            for dim in 1:3
                dof = compute_global_dof(node_id, dim, mesh)
                coords_curr[dim, local_id] += displacements[dof]
            end
        end

        Dm_inv = compute_Dm_inv(coords_ref)
        dF_dx = compute_dF_dx_tet4(Dm_inv)
        F = compute_deformation_gradient(coords_curr, coords_ref)
        dP_dF = stress_differential(material, F)
        V = compute_tet_volume(coords_ref)
        Ke = V * (dF_dx' * dP_dF * dF_dx)

        elem_dofs = get_element_dofs(mesh, element)
        for (local_i, dof_i) in enumerate(elem_dofs)
            for (local_j, dof_j) in enumerate(elem_dofs)
                if idx > length(row_indices)
                    current_len = length(row_indices)
                    growth = max(current_len >>> 1, 1)
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

    resize!(row_indices, idx - 1)
    resize!(col_indices, idx - 1)
    resize!(values, idx - 1)

    return sparse(row_indices, col_indices, values, ndof, ndof)
end

# =============================================================================
# Nonlinear Solvers - Newton-Raphson Method
# =============================================================================

"""
solve_newton(K_func, f_el_func, f_ext, active_mask; kwargs...)

Newton's method with line search for nonlinear FEM problems.

# Arguments
- `K_func`: Function that computes stiffness matrix given displacement vector
- `f_el_func`: Function that computes elastic forces given displacement vector
- `f_ext`: External force vector (full size)
- `active_mask`: Boolean mask indicating free DOFs

# Keyword Arguments
- `max_iters::Int = 1000`: Maximum Newton iterations
- `max_line_search_iters::Int = 20`: Maximum line search iterations per Newton step
- `tol::Real = 1e-4`: Convergence tolerance for residual norm
- `verbose::Bool = true`: Print iteration progress

# Returns
- Displacement vector for active DOFs
"""
function solve_newton(
    K_func::Function,
    f_el_func::Function,
    f_ext::AbstractVector,
    active_mask::BitVector;
    max_iters::Int = 1000,
    max_line_search_iters::Int = 20,
    tol::Real = 1e-4,
    verbose::Bool = true
)
    # Extract active DOFs
    f_ext_reduced = f_ext[active_mask]
    
    # Initialize solution
    U = zeros(eltype(f_ext), sum(active_mask))
    
    for iter in 1:max_iters
        # Compute stiffness matrix at current position
        K = K_func(U)
        
        # Compute elastic forces at current position
        f_el = f_el_func(U)
        
        # Compute residual
        f_res = f_ext_reduced + f_el
        f_res_norm = norm(f_res)
        
        # Check convergence
        if f_res_norm < tol
            verbose && @info "Newton's method converged in $iter iterations"
            return U
        end
        
        # Solve for search direction
        dU = K \ f_res
        
        # Line search
        step_size = 1.0
        
        for ls_iter in 1:max_line_search_iters
            # Trial solution
            U_trial = U + step_size * dU
            
            # Compute forces at trial position
            f_el_trial = f_el_func(U_trial)
            f_res_trial = f_ext_reduced + f_el_trial
            f_res_trial_norm = norm(f_res_trial)
            
            # Accept if residual decreased
            if f_res_trial_norm < f_res_norm
                U = U_trial
                verbose && @info "Iteration $iter: residual = $f_res_trial_norm, step = $step_size"
                break
            end
            
            # Halve step size
            step_size *= 0.5
            
            # Warn if line search failed
            if ls_iter == max_line_search_iters
                @warn "Line search failed at iteration $iter"
            end
        end
    end
    
    @warn "Newton's method did not converge in $max_iters iterations"
    return U
end

"""
solve_nonlinear_displacements(mesh, material, f_ext, boundary_conditions; kwargs...)

Convenience wrapper for solving nonlinear FEM problems.

# Arguments
- `mesh`: FEM mesh
- `material`: Nonlinear material model
- `f_ext`: External force vector (full size)
- `boundary_conditions`: Boolean mask for fixed nodes

# Keyword Arguments
- Passed to `solve_newton`

# Returns
- Full displacement vector
"""
function solve_nonlinear_displacements(
    mesh::Mesh{3,T,:Tet4},
    material::NonlinearMaterial,
    f_ext::AbstractVector,
    boundary_conditions::BitVector;
    kwargs...
) where T
    dofs_per_node = get_dofs_per_node(mesh)
    
    # Create DOF mask from node mask
    active_mask = BitVector(.!repeat(boundary_conditions, inner=dofs_per_node))
    active_indices = findall(active_mask)
    ndof = length(active_mask)
    
    # Define closure functions for K and f_el computation
    function K_func(U_reduced::AbstractVector)
        # Reconstruct full displacement vector
        U_full = zeros(T, ndof)
        U_full[active_indices] = U_reduced
        
        # Assemble consistent tangent stiffness
        K_full = assemble_nonlinear_tangent(mesh, material, U_full)
        return K_full[active_indices, active_indices]
    end
    
    function f_el_func(U_reduced::AbstractVector)
        # Reconstruct full displacement vector
        U_full = zeros(T, ndof)
        U_full[active_indices] = U_reduced
        
        # Compute elastic forces
        f_el_full = compute_elastic_forces(mesh, material, U_full)
        return f_el_full[active_indices]
    end
    
    # Solve using Newton's method
    U_reduced = solve_newton(K_func, f_el_func, f_ext, active_mask; kwargs...)
    
    # Reconstruct full displacement vector
    U_full = zeros(T, length(active_mask))
    U_full[active_indices] = U_reduced
    
    return U_full
end
