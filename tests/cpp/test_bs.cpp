// Black-Scholes unit tests. Every expected value is derived independently of
// the implementation: published reference prices, put-call parity, or
// analytic limits. Tolerances are stated next to each check.
#include <cmath>

#include "optlib/black_scholes.hpp"
#include "test_main.hpp"

using optlib::Market;
using optlib::OptionType;

namespace {
const Market kRef{100.0, 100.0, 0.05, 0.0, 0.2, 1.0};
double call(const Market& m) { return optlib::bs_price(OptionType::Call, m); }
double put(const Market& m) { return optlib::bs_price(OptionType::Put, m); }
}  // namespace

// Reference case from the README; values are quoted to 4 decimals, so the
// tolerance is half a unit in the last quoted digit (5e-5).
TEST(reference_case_matches_quoted_values) {
    CHECK_NEAR(call(kRef), 10.4506, 5e-5);
    CHECK_NEAR(put(kRef), 5.5735, 5e-5);
}

// Hull, "Options, Futures and Other Derivatives", worked example:
// S=42, K=40, r=10%, sigma=20%, T=0.5 -> c = 4.76, p = 0.81 (2 decimals quoted,
// so tolerance 5e-3).
TEST(hull_textbook_example) {
    const Market m{42.0, 40.0, 0.10, 0.0, 0.20, 0.5};
    CHECK_NEAR(call(m), 4.76, 5e-3);
    CHECK_NEAR(put(m), 0.81, 5e-3);
}

// Put-call parity C - P = S e^{-qT} - K e^{-rT} is model-free. Tolerance 1e-12
// relative to S: only floating-point rounding separates the two sides.
TEST(put_call_parity_with_dividends) {
    const Market cases[] = {
        {100, 100, 0.05, 0.00, 0.2, 1.0}, {100, 80, 0.03, 0.02, 0.35, 2.0},
        {50, 70, 0.01, 0.04, 0.15, 0.25}, {120, 100, -0.01, 0.0, 0.5, 5.0},
    };
    for (const auto& m : cases) {
        const double rhs = m.S * std::exp(-m.q * m.T) - m.K * std::exp(-m.r * m.T);
        CHECK_NEAR(call(m) - put(m), rhs, 1e-12 * m.S);
    }
}

// A dividend yield q is equivalent to replacing spot by S e^{-qT} with q = 0.
// Tolerance 1e-12: identical math up to rounding.
TEST(dividend_yield_equals_spot_adjustment) {
    const Market m{100, 95, 0.04, 0.03, 0.25, 1.5};
    Market adj = m;
    adj.S = m.S * std::exp(-m.q * m.T);
    adj.q = 0.0;
    CHECK_NEAR(call(m), call(adj), 1e-12 * m.S);
    CHECK_NEAR(put(m), put(adj), 1e-12 * m.S);
}

// Analytic limits (exact by definition, tolerance 1e-12).
TEST(degenerate_limits) {
    Market expiry = kRef;
    expiry.T = 0.0;
    expiry.S = 110.0;
    CHECK_NEAR(call(expiry), 10.0, 1e-12);
    CHECK_NEAR(put(expiry), 0.0, 1e-12);

    Market novol = kRef;
    novol.sigma = 0.0;
    const double fwd_intrinsic = novol.S - novol.K * std::exp(-novol.r * novol.T);
    CHECK_NEAR(call(novol), fwd_intrinsic, 1e-12);
    CHECK_NEAR(put(novol), 0.0, 1e-12);
}

// Continuity at the limits: tiny sigma / tiny T should approach the limit
// values. Tolerance 1e-6: N(d) saturates to 0/1 exponentially fast here.
TEST(limits_are_continuous) {
    Market m = kRef;
    m.sigma = 1e-6;
    CHECK_NEAR(call(m), m.S - m.K * std::exp(-m.r * m.T), 1e-6);
    m = kRef;
    m.S = 110.0;
    m.T = 1e-10;
    CHECK_NEAR(call(m), 10.0, 1e-6);
}

// Deep in/out of the money: call -> S e^{-qT} - K e^{-rT}, put -> 0.
// With S/K = 10 and sigma*sqrt(T)=0.2, N(-d2) ~ 1e-30, so tolerance 1e-10.
TEST(deep_moneyness_limits) {
    Market m{1000, 100, 0.05, 0.01, 0.2, 1.0};
    CHECK_NEAR(call(m), m.S * std::exp(-m.q * m.T) - m.K * std::exp(-m.r * m.T), 1e-10);
    CHECK_NEAR(put(m), 0.0, 1e-10);
}

// Price bounds: max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}.
TEST(no_arbitrage_bounds) {
    const Market m{100, 105, 0.05, 0.02, 0.3, 2.0};
    const double lo = std::max(m.S * std::exp(-m.q * m.T) - m.K * std::exp(-m.r * m.T), 0.0);
    CHECK(call(m) >= lo);
    CHECK(call(m) <= m.S * std::exp(-m.q * m.T));
    CHECK(put(m) <= m.K * std::exp(-m.r * m.T));
}

// Greek identities implied by parity (independent of the Greek formulas):
//   delta_c - delta_p = e^{-qT},  gamma_c = gamma_p,  vega_c = vega_p,
//   rho_c - rho_p = K T e^{-rT},  theta_c - theta_p = q S e^{-qT} - r K e^{-rT}.
// Tolerance 1e-12 (rounding only).
TEST(greek_parity_identities) {
    const Market m{100, 90, 0.04, 0.02, 0.3, 1.25};
    const auto c = optlib::bs_greeks(OptionType::Call, m);
    const auto p = optlib::bs_greeks(OptionType::Put, m);
    const double dq = std::exp(-m.q * m.T), dr = std::exp(-m.r * m.T);
    CHECK_NEAR(c.delta - p.delta, dq, 1e-12);
    CHECK_NEAR(c.gamma, p.gamma, 1e-12);
    CHECK_NEAR(c.vega, p.vega, 1e-12);
    CHECK_NEAR(c.rho - p.rho, m.K * m.T * dr, 1e-10);
    CHECK_NEAR(c.theta - p.theta, m.q * m.S * dq - m.r * m.K * dr, 1e-10);
}

TEST(greek_signs) {
    const auto c = optlib::bs_greeks(OptionType::Call, kRef);
    const auto p = optlib::bs_greeks(OptionType::Put, kRef);
    CHECK(c.delta > 0 && c.delta < 1);
    CHECK(p.delta < 0 && p.delta > -1);
    CHECK(c.gamma > 0 && c.vega > 0);
    CHECK(c.theta < 0);
    CHECK(c.rho > 0 && p.rho < 0);
}

// Greeks vs central finite differences of the price. Central differences have
// O(h^2) truncation error and O(eps*V/h) rounding error (O(eps*V/h^2) for
// gamma); with these step sizes both are < ~1e-6, so tolerance 1e-5.
TEST(greeks_match_finite_differences) {
    const Market base{100, 95, 0.05, 0.02, 0.25, 1.5};
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        const auto g = optlib::bs_greeks(type, base);
        auto price = [&](Market m) { return optlib::bs_price(type, m); };

        const double hS = 1e-3 * base.S;
        Market up = base, dn = base;
        up.S += hS;
        dn.S -= hS;
        CHECK_NEAR(g.delta, (price(up) - price(dn)) / (2 * hS), 1e-5);
        CHECK_NEAR(g.gamma, (price(up) - 2 * price(base) + price(dn)) / (hS * hS), 1e-5);

        const double hv = 1e-4;
        up = base; dn = base;
        up.sigma += hv; dn.sigma -= hv;
        CHECK_NEAR(g.vega, (price(up) - price(dn)) / (2 * hv), 1e-5);

        const double hr = 1e-4;
        up = base; dn = base;
        up.r += hr; dn.r -= hr;
        CHECK_NEAR(g.rho, (price(up) - price(dn)) / (2 * hr), 1e-5);

        // theta = dV/dt = -dV/dT
        const double hT = 1e-4;
        up = base; dn = base;
        up.T += hT; dn.T -= hT;
        CHECK_NEAR(g.theta, -(price(up) - price(dn)) / (2 * hT), 1e-5);
    }
}

TEST(invalid_inputs_throw) {
    Market m = kRef;
    m.S = -1;
    CHECK_THROWS(call(m));
    m = kRef;
    m.sigma = -0.1;
    CHECK_THROWS(call(m));
    m = kRef;
    m.T = -1;
    CHECK_THROWS(call(m));
    m = kRef;
    m.K = std::nan("");
    CHECK_THROWS(call(m));
    m = kRef;
    m.T = 0;
    CHECK_THROWS(optlib::bs_greeks(OptionType::Call, m));
}

TEST_MAIN()
