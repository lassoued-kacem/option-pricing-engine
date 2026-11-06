#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace optlib {

// Thomas algorithm for a tridiagonal system
//   lower[i] x[i-1] + diag[i] x[i] + upper[i] x[i+1] = rhs[i],  i = 0..n-1
// (lower[0] and upper[n-1] are ignored). O(n). Stable without pivoting for
// diagonally dominant matrices, which the Crank-Nicolson / implicit matrices
// I - theta dt L are whenever dx is small enough that the convection term does
// not dominate diffusion. Solves in place: `rhs` is overwritten with x.
inline void solve_tridiagonal(const std::vector<double>& lower, const std::vector<double>& diag,
                              const std::vector<double>& upper, std::vector<double>& rhs,
                              std::vector<double>& scratch) {
    const std::size_t n = diag.size();
    if (n == 0 || lower.size() != n || upper.size() != n || rhs.size() != n)
        throw std::invalid_argument("solve_tridiagonal: size mismatch");
    scratch.resize(n);

    double denom = diag[0];
    if (denom == 0.0) throw std::runtime_error("solve_tridiagonal: zero pivot");
    scratch[0] = upper[0] / denom;
    rhs[0] /= denom;
    for (std::size_t i = 1; i < n; ++i) {
        denom = diag[i] - lower[i] * scratch[i - 1];
        if (denom == 0.0) throw std::runtime_error("solve_tridiagonal: zero pivot");
        scratch[i] = upper[i] / denom;
        rhs[i] = (rhs[i] - lower[i] * rhs[i - 1]) / denom;
    }
    for (std::size_t i = n - 1; i-- > 0;) rhs[i] -= scratch[i] * rhs[i + 1];
}

}  // namespace optlib
