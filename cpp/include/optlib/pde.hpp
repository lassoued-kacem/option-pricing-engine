#pragma once

#include <cstddef>

#include "optlib/types.hpp"

namespace optlib {

// Crank-Nicolson finite-difference price of a European option.
//
// Solves the Black-Scholes PDE in log-spot x = ln S and time-to-maturity tau:
//   V_tau = 1/2 sigma^2 V_xx + (r - q - 1/2 sigma^2) V_x - r V,
//   V(x, 0) = payoff(e^x),
// on a uniform grid x in [ln K - L, ln K + L] with
// L = |ln(S/K)| + width_sd * sigma * sqrt(T) (default width_sd = 6),
// so the strike (the payoff kink) sits exactly on the middle node. Dirichlet
// boundaries use the exact asymptotic solutions (linear in S):
//   call: V(S_min) = 0,                        V(S_max) = S_max e^{-q tau} - K e^{-r tau}
//   put:  V(S_min) = K e^{-r tau} - S_min e^{-q tau},  V(S_max) = 0
// The first time step is replaced by two fully implicit half-steps (Rannacher
// start), which damps the oscillations CN otherwise produces from the payoff
// kink and restores clean second-order convergence. The price at S is read off
// by cubic Lagrange interpolation on the four nearest nodes.
//
//   n_space: number of space intervals (even, >= 4)
//   n_time:  number of time steps (>= 2)
//   width_sd: domain half-width beyond |ln(S/K)|, in units of sigma sqrt(T) (> 0)
// Requires sigma > 0. T == 0 returns the payoff.
double pde_price(OptionType type, const Market& m, std::size_t n_space = 400,
                 std::size_t n_time = 400, double width_sd = 6.0);

}  // namespace optlib
