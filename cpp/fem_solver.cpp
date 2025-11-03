#include "fem_solver.hpp"
#include <Eigen/Sparse> // just use Eigen library for this draft implementation
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <memory>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>

namespace {
thread_local std::string g_last_error;

struct FactorImpl {
    using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>;
    SpMat A;
    std::unique_ptr<Eigen::SimplicialLDLT<SpMat>> ldlt;
    std::unique_ptr<Eigen::SparseLU<SpMat>> lu;
    bool isSpd{true};
};

void set_error(const std::string& msg) { g_last_error = msg; }
} // namespace

extern "C" {

struct fem_factor_handle {
    std::unique_ptr<FactorImpl> impl;
};

fem_factor_handle* fem_factor_create(
    std::int64_t n, std::int64_t nnz,
    const std::int64_t* colptr, const std::int64_t* rowidx, const double* values,
    int is_spd)
{
    try {
        auto handle = std::make_unique<fem_factor_handle>();
        handle->impl = std::make_unique<FactorImpl>();
        handle->impl->isSpd = (is_spd != 0);
        
        using SpMat = FactorImpl::SpMat;
        SpMat& A = handle->impl->A;
        A.resize(n, n);
        
        std::vector<Eigen::Triplet<double, std::int64_t>> trips;
        trips.reserve(nnz);
        
        for (std::int64_t j = 0; j < n; ++j) {
            for (std::int64_t k = colptr[j] - 1; k < colptr[j + 1] - 1; ++k) {
                trips.emplace_back(rowidx[k] - 1, j, values[k]);
            }
        }
        
        A.setFromTriplets(trips.begin(), trips.end());
        
        if (handle->impl->isSpd) {
            handle->impl->ldlt = std::make_unique<Eigen::SimplicialLDLT<SpMat>>();
            handle->impl->ldlt->compute(A);
        } else {
            handle->impl->lu = std::make_unique<Eigen::SparseLU<SpMat>>();
            handle->impl->lu->compute(A);
        }
        
        return handle.release();
    } catch (...) {
        set_error("Factorization failed");
        return nullptr;
    }
}

int fem_solve(fem_factor_handle* handle, const double* b, double* x)
{
    try {
        const auto n = handle->impl->A.rows();
        Eigen::Map<const Eigen::VectorXd> bvec(b, n);
        Eigen::Map<Eigen::VectorXd> xvec(x, n);
        
        if (handle->impl->isSpd && handle->impl->ldlt) {
            xvec = handle->impl->ldlt->solve(bvec);
        } else if (handle->impl->lu) {
            xvec = handle->impl->lu->solve(bvec);
        } else {
            return -1;
        }
        return 0;
    } catch (...) {
        set_error("Solve failed");
        return -1;
    }
}

void fem_factor_free(fem_factor_handle* handle)
{
    delete handle;
}

std::size_t fem_last_error(char* buffer, std::size_t buflen)
{
    if (!buffer || buflen == 0) return 0;
        std::size_t n = std::min(buflen - 1, g_last_error.size());
        std::memcpy(buffer, g_last_error.data(), n);
        buffer[n] = '\0';
        return n;
}

} // extern "C"