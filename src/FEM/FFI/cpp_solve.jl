# FFI submodule for C++ linear solver integration  
module FFI

using SparseArrays, LinearAlgebra

"""
    cpp_solve(K, f; is_spd=true)

Solve K \\ f using the C++ backend. 
Currently falls back to Julia's standard solver until C++ library is implemented.
"""
function cpp_solve(K::SparseMatrixCSC, f::AbstractVector; is_spd::Bool=true)
    # TODO: Once C++ library is built, implement actual FFI calls here
    # For now, fallback to standard Julia solver
    return K \ f
end

end # module FFI