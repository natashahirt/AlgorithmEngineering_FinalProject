# solvers.jl - FEM solvers for linear FEM problems
"""
Linear solvers for finite element analysis.
"""

"""
solve_displacements(K, f)

Solve the linear system `K * u = f` for the displacement vector.
Uses C++ solver by default with fallback to Julia's standard solver if needed.
"""
function solve_displacements(K, f; solver::Symbol=:cpp, is_spd::Bool=true)
    solver in [:standard, :cpp] || throw(ArgumentError("Solver must be either :standard (default) or :cpp, got :$solver"))
    try
        if solver == :standard
            return K \ f
        elseif solver == :cpp
            out = FEM.FFI.cpp_solve(K, f; is_spd=is_spd)
            return out
        end
    catch e
        @warn "C++ solver failed, falling back to Julia standard solver: $e"
        return K \ f  # Fallback to Julia's built-in sparse solver
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
