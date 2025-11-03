#pragma once
#include <cstddef>
#include <cstdint>

// avoid linker mangling
#ifdef __cplusplus
extern "C" {
#endif

// factorization from Julia SparseMatrixCSC
// used for cholesky or LU to solve linear systems

typedef struct fem_factor_handle fem_factor_handle;
fem_factor_handle* fem_factor_create(
    std::int64_t n, // dimension
    std::int64_t nnz, // number of nonzeros
    const std::int64_t* colptr, // size n+1 (1-based from Julia, make sure to shift to 0-based)
    const std::int64_t* rowidx, // size nnz (also 1-based)
    const double* values, // size nnz
    int is_spd // if nonzero, then is spd and we need to use SimplicialLDLT. else use SparseLU
);

// solve Ax = b
// if successful return 0
int fem_solve(
    fem_factor_handle* handle,
    const double* b, // size n
    double* x // output, size n
);

// free up some resources
void fem_factor_free(fem_factor_handle* handle);

// get the last error message
std::size_t fem_last_error(char* buffer, std::size_t buflen);

#ifdef __cplusplus
} // extern "C"
#endif