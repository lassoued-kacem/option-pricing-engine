#pragma once

#include <cstddef>
#include <cstdint>

#include "optlib/types.hpp"

namespace optlib {

// A Monte Carlo estimate always carries its standard error.
struct McResult {
    double price;      // discounted mean payoff
    double std_error;  // standard error of `price` (sample std / sqrt(#i.i.d. samples))
    std::size_t n_paths;  // number of simulated terminal prices
};

// All estimators sample the exact GBM terminal distribution under the
// risk-neutral measure (no time stepping is needed for European payoffs):
//   S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0,1)
// and discount the payoff by e^{-rT}. Random numbers come from a seeded
// std::mt19937_64, so results are reproducible for a given seed.

// Plain estimator: n_paths independent draws. Baseline for variance comparisons.
McResult mc_plain(OptionType type, const Market& m, std::size_t n_paths, std::uint64_t seed);

// Antithetic variates: n_paths/2 pairs (Z, -Z). Each pair's average payoff is
// one i.i.d. sample, so the standard error is computed over n_paths/2 pair
// means. n_paths must be even and >= 4.
McResult mc_antithetic(OptionType type, const Market& m, std::size_t n_paths,
                       std::uint64_t seed);

// Antithetic variates plus a control variate. Control X = e^{-rT} S_T with known
// mean S e^{-qT}. Working on pair means Y_i (payoff) and X_i (control):
//   price = mean(Y) - beta (mean(X) - S e^{-qT}),  beta = Cov(Y,X) / Var(X)
// beta is estimated from the same samples, which adds an O(1/n) bias that is
// negligible next to the O(1/sqrt(n)) standard error. The reported standard
// error is std(Y - beta X) / sqrt(n_pairs), with n_pairs - 2 degrees of
// freedom. n_paths must be even and >= 6 (at least 3 pairs).
// Limitation: if almost no simulated path lands on one side of the strike
// (very deep in/out of the money relative to sigma sqrt(T) and n_paths), the
// payoff is linear in S_T on every sampled path, the control absorbs it
// completely and the reported SE collapses towards 0 although the true error
// (from the unsampled region) does not. Treat a near-zero SE with suspicion.
McResult mc_control_variate(OptionType type, const Market& m, std::size_t n_paths,
                            std::uint64_t seed);

}  // namespace optlib
