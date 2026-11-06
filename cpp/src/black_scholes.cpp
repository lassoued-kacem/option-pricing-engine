#include "optlib/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace optlib {

namespace {
constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr double kInvSqrt2Pi = 0.39894228040143267794;
}  // namespace

void validate(const Market& m) {
    const bool finite = std::isfinite(m.S) && std::isfinite(m.K) && std::isfinite(m.r) &&
                        std::isfinite(m.q) && std::isfinite(m.sigma) && std::isfinite(m.T);
    if (!finite) throw std::invalid_argument("market parameters must be finite");
    if (m.S <= 0.0) throw std::invalid_argument("spot S must be > 0");
    if (m.K <= 0.0) throw std::invalid_argument("strike K must be > 0");
    if (m.sigma < 0.0) throw std::invalid_argument("volatility sigma must be >= 0");
    if (m.T < 0.0) throw std::invalid_argument("maturity T must be >= 0");
}

// erfc form keeps full relative precision in the far left tail.
double norm_cdf(double x) { return 0.5 * std::erfc(-x * kInvSqrt2); }

double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

double bs_price(OptionType type, const Market& m) {
    validate(m);
    const double sign = (type == OptionType::Call) ? 1.0 : -1.0;

    if (m.T == 0.0) return std::max(sign * (m.S - m.K), 0.0);

    const double df_r = std::exp(-m.r * m.T);
    const double df_q = std::exp(-m.q * m.T);
    if (m.sigma == 0.0) return std::max(sign * (m.S * df_q - m.K * df_r), 0.0);

    const double vol_sqrt_t = m.sigma * std::sqrt(m.T);
    const double d1 =
        (std::log(m.S / m.K) + (m.r - m.q + 0.5 * m.sigma * m.sigma) * m.T) / vol_sqrt_t;
    const double d2 = d1 - vol_sqrt_t;

    // Call: S e^{-qT} N(d1) - K e^{-rT} N(d2)
    // Put:  K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    // Clamp: far out of the money the difference can round to -0.0 or a
    // negative subnormal; a European option value is never negative.
    return std::max(
        sign * (m.S * df_q * norm_cdf(sign * d1) - m.K * df_r * norm_cdf(sign * d2)), 0.0);
}

Greeks bs_greeks(OptionType type, const Market& m) {
    validate(m);
    if (m.T <= 0.0 || m.sigma <= 0.0)
        throw std::invalid_argument("Greeks require T > 0 and sigma > 0");

    const double sign = (type == OptionType::Call) ? 1.0 : -1.0;
    const double sqrt_t = std::sqrt(m.T);
    const double vol_sqrt_t = m.sigma * sqrt_t;
    const double d1 =
        (std::log(m.S / m.K) + (m.r - m.q + 0.5 * m.sigma * m.sigma) * m.T) / vol_sqrt_t;
    const double d2 = d1 - vol_sqrt_t;
    const double df_r = std::exp(-m.r * m.T);
    const double df_q = std::exp(-m.q * m.T);
    const double pdf_d1 = norm_pdf(d1);
    const double Nd1 = norm_cdf(sign * d1);  // N(d1) for call, N(-d1) for put
    const double Nd2 = norm_cdf(sign * d2);  // N(d2) for call, N(-d2) for put

    Greeks g{};
    g.delta = sign * df_q * Nd1;
    g.gamma = df_q * pdf_d1 / (m.S * vol_sqrt_t);
    g.vega = m.S * df_q * pdf_d1 * sqrt_t;
    // theta = dV/dt = -dV/dT
    g.theta = -m.S * df_q * pdf_d1 * m.sigma / (2.0 * sqrt_t)
              - sign * m.r * m.K * df_r * Nd2
              + sign * m.q * m.S * df_q * Nd1;
    g.rho = sign * m.K * m.T * df_r * Nd2;
    return g;
}

}  // namespace optlib
