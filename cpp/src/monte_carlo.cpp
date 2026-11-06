#include "optlib/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace optlib {

namespace {

// Welford running mean/variance (numerically stable single pass).
struct Moments {
    std::size_t n = 0;
    double mean = 0.0;
    double m2 = 0.0;
    void add(double x) {
        ++n;
        const double d = x - mean;
        mean += d / static_cast<double>(n);
        m2 += d * (x - mean);
    }
    double variance() const { return n > 1 ? m2 / static_cast<double>(n - 1) : 0.0; }
};

// Running means and co-moments of (Y, X) for the control-variate estimator.
struct CoMoments {
    std::size_t n = 0;
    double my = 0.0, mx = 0.0;
    double syy = 0.0, sxx = 0.0, sxy = 0.0;
    void add(double y, double x) {
        ++n;
        const double inv = 1.0 / static_cast<double>(n);
        const double dx = x - mx;
        const double dy = y - my;
        mx += dx * inv;
        my += dy * inv;
        // Co-moment update uses the old delta of one variable and the new
        // delta of the other (standard Welford/Chan form).
        sxx += dx * (x - mx);
        syy += dy * (y - my);
        sxy += dx * (y - my);
    }
};

struct GbmTerminal {
    double S0, drift_t, vol_sqrt_t;
    double operator()(double z) const { return S0 * std::exp(drift_t + vol_sqrt_t * z); }
};

GbmTerminal make_gbm(const Market& m) {
    return {m.S, (m.r - m.q - 0.5 * m.sigma * m.sigma) * m.T, m.sigma * std::sqrt(m.T)};
}

inline double payoff(OptionType type, double ST, double K) {
    return type == OptionType::Call ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0);
}

void check_paths(std::size_t n_paths, bool paired) {
    if (n_paths < 4) throw std::invalid_argument("n_paths must be >= 4");
    if (paired && n_paths % 2 != 0)
        throw std::invalid_argument("n_paths must be even for antithetic sampling");
}

}  // namespace

McResult mc_plain(OptionType type, const Market& m, std::size_t n_paths, std::uint64_t seed) {
    validate(m);
    check_paths(n_paths, false);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto gbm = make_gbm(m);
    const double df = std::exp(-m.r * m.T);

    Moments acc;
    for (std::size_t i = 0; i < n_paths; ++i) acc.add(df * payoff(type, gbm(normal(rng)), m.K));
    return {acc.mean, std::sqrt(acc.variance() / static_cast<double>(acc.n)), n_paths};
}

McResult mc_antithetic(OptionType type, const Market& m, std::size_t n_paths,
                       std::uint64_t seed) {
    validate(m);
    check_paths(n_paths, true);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto gbm = make_gbm(m);
    const double df = std::exp(-m.r * m.T);

    Moments acc;  // over pair means
    const std::size_t n_pairs = n_paths / 2;
    for (std::size_t i = 0; i < n_pairs; ++i) {
        const double z = normal(rng);
        const double y = 0.5 * df * (payoff(type, gbm(z), m.K) + payoff(type, gbm(-z), m.K));
        acc.add(y);
    }
    return {acc.mean, std::sqrt(acc.variance() / static_cast<double>(acc.n)), n_paths};
}

McResult mc_control_variate(OptionType type, const Market& m, std::size_t n_paths,
                            std::uint64_t seed) {
    validate(m);
    check_paths(n_paths, true);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto gbm = make_gbm(m);
    const double df = std::exp(-m.r * m.T);
    const double control_mean = m.S * std::exp(-m.q * m.T);  // E[e^{-rT} S_T]

    CoMoments acc;
    const std::size_t n_pairs = n_paths / 2;
    for (std::size_t i = 0; i < n_pairs; ++i) {
        const double z = normal(rng);
        const double s_up = gbm(z), s_dn = gbm(-z);
        const double y = 0.5 * df * (payoff(type, s_up, m.K) + payoff(type, s_dn, m.K));
        const double x = 0.5 * df * (s_up + s_dn);
        acc.add(y, x);
    }

    const double n = static_cast<double>(acc.n);
    // Degenerate control (sigma = 0 or T = 0): X is constant, no correction.
    const double beta = acc.sxx > 0.0 ? acc.sxy / acc.sxx : 0.0;
    const double price = acc.my - beta * (acc.mx - control_mean);
    // Var(Y - beta X) = Var(Y) - 2 beta Cov(Y,X) + beta^2 Var(X); n - 2 dof
    // because the mean and beta are both estimated.
    const double resid_ss = std::max(acc.syy - 2.0 * beta * acc.sxy + beta * beta * acc.sxx, 0.0);
    const double resid_var = resid_ss / (n - 2.0);
    return {price, std::sqrt(resid_var / n), n_paths};
}

}  // namespace optlib
