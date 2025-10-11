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
