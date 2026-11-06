#pragma once

#include "optlib/types.hpp"

namespace optlib {

// Standard normal CDF and PDF.
double norm_cdf(double x);
double norm_pdf(double x);

// Black-Scholes-Merton price of a European option with continuous dividend
// yield q. Degenerate inputs are handled by their analytic limits:
//   T == 0      -> intrinsic value max(+-(S - K), 0)
//   sigma == 0  -> discounted forward intrinsic max(+-(S e^{-qT} - K e^{-rT}), 0)
double bs_price(OptionType type, const Market& m);

// Analytic Greeks. Conventions:
//   delta = dV/dS, gamma = d2V/dS2
//   vega  = dV/dsigma per 1.00 change in sigma (not per 1%)
//   theta = dV/dt in calendar time, per year (= -dV/dT)
//   rho   = dV/dr per 1.00 change in r (not per 1%)
// Requires T > 0 and sigma > 0 (throws std::invalid_argument otherwise).
struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

Greeks bs_greeks(OptionType type, const Market& m);

}  // namespace optlib
