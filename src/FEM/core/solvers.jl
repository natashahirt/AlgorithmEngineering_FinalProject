# solvers.jl - FEM solvers for linear FEM problems
"""
Linear solvers for finite element analysis.
"""

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

function solve_fem(K::AbstractMatrix{T}, f::AbstractVector{T}, boundary_dofs::AbstractVector{<:Integer}; prescribed_values::Union{Nothing, AbstractVector{T}}=nothing) where {T}

    if isnothing(prescribed_values)
        prescribed_values = zeros(T, length(boundary_dofs))
    end

    K_mod, f_mod = apply_boundary_conditions(K, f, boundary_dofs, prescribed_values) # Apply Dirichlet boundary conditions (zero out the fixed DOFs)
    
    U = solve_displacements(K_mod, f_mod)

    return U

end