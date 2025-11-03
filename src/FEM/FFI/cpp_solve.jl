module FFI

using SparseArrays

# ---- platform ext without Libdl ----
const _LIBEXT  = Sys.isapple() ? "dylib" : Sys.islinux() ? "so" : "dll"
const _LIBNAME = "libfemsolver." * _LIBEXT
const _LIBPATH = get(ENV, "JULIA_FEMSOLVER_LIB",
                     abspath(joinpath(@__DIR__, "..", "..", "..", "cpp", "build", _LIBNAME)))

@noinline function _ensure_lib()
    isfile(_LIBPATH) || error("C++ solver library not found at: $_LIBPATH\n" *
                              "Set JULIA_FEMSOLVER_LIB or build it:\n" *
                              "  cd cpp && mkdir -p build && cd build && cmake .. && cmake --build .")
    nothing
end

# pull last error string from C++
@noinline function _last_error()::String
    _ensure_lib()
    buf = Vector{UInt8}(undef, 512)
    n = ccall((:fem_last_error, _LIBPATH), Csize_t,
              (Ptr{Cchar}, Csize_t), pointer(buf), length(buf))
    String(resize!(buf, Int(n)))
end

mutable struct FactorHandle
    ptr::Ptr{Cvoid}
end
function Base.finalize(h::FactorHandle)
    if h.ptr != C_NULL
        ccall((:fem_factor_free, _LIBPATH), Cvoid, (Ptr{Cvoid},), h.ptr)
        h.ptr = C_NULL
    end
    nothing
end

"""
    cpp_solve(K::SparseMatrixCSC{Float64,Int64}, f::AbstractVector{Float64}; is_spd=true)

Solve `K \\ f` via the C++ backend, assuming SPD if `is_spd=true`.
"""
function cpp_solve(K::SparseMatrixCSC{Float64,Int64},
                   f::AbstractVector{Float64}; is_spd::Bool=true)
    _ensure_lib()

    n = size(K,1)
    size(K,2) == n || throw(ArgumentError("K must be square"))
    length(f) == n || throw(DimensionMismatch("length(f) != size(K,1)"))

    b = Vector{Float64}(f)
    x = similar(b)

    colptr = K.colptr
    rowind = K.rowval
    vals   = K.nzval

    GC.@preserve colptr rowind vals b x begin
        handle_ptr = ccall((:fem_factor_create, _LIBPATH), Ptr{Cvoid},
                           (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Cint),
                           n, nnz(K), pointer(colptr), pointer(rowind), pointer(vals),
                           is_spd ? 1 : 0)
        handle_ptr == C_NULL && error("fem_factor_create failed: " * _last_error())
        handle = FactorHandle(handle_ptr)
        try
            rc = ccall((:fem_solve, _LIBPATH), Cint,
                       (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}),
                       handle.ptr, pointer(b), pointer(x))
            rc == 0 || error("fem_solve failed (code $rc): " * _last_error())
        finally
            finalize(handle)
        end
    end
    return x
end

# precise type error if someone passes the wrong matrix eltypes
function cpp_solve(K::SparseMatrixCSC, f::AbstractVector; is_spd::Bool=true)
    eltype(K) === Float64 || throw(ArgumentError("K must have Float64 values"))
    K.rowval isa Vector{Int64} || throw(ArgumentError("K must use Int64 indices"))
    throw(ArgumentError("K must be SparseMatrixCSC{Float64,Int64}"))
end

end # module
