// Crank-Nicolson PDE unit tests. Ground truth is the Black-Scholes closed form.
#include <cmath>
#include <cstddef>
#include <initializer_list>

#include "optlib/black_scholes.hpp"
#include "optlib/pde.hpp"
#include "optlib/tridiag.hpp"
#include "test_main.hpp"

using optlib::Market;
using optlib::OptionType;

namespace {
const Market kRef{100, 100, 0.05, 0.0, 0.2, 1.0};
const Market kCases[] = {
    kRef,
    {100, 120, 0.03, 0.02, 0.30, 2.0},  // dividends, OTM call
    {80, 70, 0.01, 0.04, 0.15, 0.5},    // q > r
    {90, 100, 0.05, 0.00, 0.60, 3.0},   // high vol, long dated (coarsest grid)
};
}  // namespace

// Thomas solver against a hand-solvable 3x3 system:
//   [2 1 0; 1 2 1; 0 1 2] x = [4 8 8]  ->  x = [1 2 3]  (checked by substitution).
TEST(tridiagonal_solver_known_system) {
    std::vector<double> lo{0, 1, 1}, di{2, 2, 2}, up{1, 1, 0}, rhs{4, 8, 8}, scratch;
    optlib::solve_tridiagonal(lo, di, up, rhs, scratch);
    CHECK_NEAR(rhs[0], 1.0, 1e-14);
    CHECK_NEAR(rhs[1], 2.0, 1e-14);
    CHECK_NEAR(rhs[2], 3.0, 1e-14);
}

// Reference case at the default 400 x 400 grid: required accuracy 1e-3
// absolute (0.01% of the price), the plan's accuracy target.
TEST(reference_case_default_grid) {
    CHECK_NEAR(optlib::pde_price(OptionType::Call, kRef), optlib::bs_price(OptionType::Call, kRef),
               1e-3);
    CHECK_NEAR(optlib::pde_price(OptionType::Put, kRef), optlib::bs_price(OptionType::Put, kRef),
               1e-3);
}

// All cases at 800 x 800: relative error below 1e-4 of the price (floor 1e-4
// absolute for cheap options). The grid half-width scales with sigma*sqrt(T),
// so dx is ~5x larger for the sigma=0.6, T=3 case; a relative target is the
// meaningful accuracy spec across cases.
TEST(all_cases_fine_grid) {
    for (const auto& m : kCases)
        for (OptionType t : {OptionType::Call, OptionType::Put}) {
            const double bs = optlib::bs_price(t, m);
            CHECK_NEAR(optlib::pde_price(t, m, 800, 800), bs, 1e-4 * std::fmax(bs, 1.0));
        }
}

// Second-order convergence: CN with a Rannacher start has error C(dx^2 + dt^2).
// Doubling Nx and Nt together must divide the error by ~4. Band [3.5, 4.5]
// allows for the higher-order terms still present at Nx = 100..400.
TEST(second_order_convergence) {
    for (const auto& m : kCases)
        for (OptionType t : {OptionType::Call, OptionType::Put}) {
            const double bs = optlib::bs_price(t, m);
            const double e1 = optlib::pde_price(t, m, 100, 100) - bs;
            const double e2 = optlib::pde_price(t, m, 200, 200) - bs;
            const double e3 = optlib::pde_price(t, m, 400, 400) - bs;
            const double r1 = e1 / e2, r2 = e2 / e3;
            CHECK(r1 > 3.5 && r1 < 4.5);
            CHECK(r2 > 3.5 && r2 < 4.5);
        }
}

// Put-call parity holds for the PDE only up to discretisation error, so the
// tolerance matches the fine-grid accuracy target above (two prices' worth).
TEST(put_call_parity) {
    for (const auto& m : kCases) {
        const double c = optlib::pde_price(OptionType::Call, m, 800, 800);
        const double p = optlib::pde_price(OptionType::Put, m, 800, 800);
        const double rhs = m.S * std::exp(-m.q * m.T) - m.K * std::exp(-m.r * m.T);
        CHECK_NEAR(c - p, rhs, 2e-4 * std::fmax(m.S, 1.0));
    }
}

// Boundary conditions. At the default 6-sd domain the boundaries barely
// influence the price at S, so a wrong boundary value would go unnoticed.
// On a narrow 2-sd domain they matter: the linear asymptotes used as boundary
// values are still close to the true solution there (the option's time value
// at 2 sd is small), so a correct implementation stays within 2e-3 absolute,
// while a boundary off by O(K r T) moves the price by ~1e-2 or more.
TEST(boundary_conditions_on_narrow_domain) {
    for (const auto& m : kCases)
        for (OptionType t : {OptionType::Call, OptionType::Put}) {
            const double bs = optlib::bs_price(t, m);
            CHECK_NEAR(optlib::pde_price(t, m, 800, 800, 2.0), bs, 2e-3);
        }
}

// Spot away from a node (interpolation path) and away from the strike.
TEST(off_node_spot) {
    const Market m{103.7, 100, 0.05, 0.01, 0.25, 0.75};
    for (OptionType t : {OptionType::Call, OptionType::Put}) {
        const double bs = optlib::bs_price(t, m);
        CHECK_NEAR(optlib::pde_price(t, m, 800, 800), bs, 1e-4 * std::fmax(bs, 1.0));
    }
}

TEST(expiry_returns_payoff_and_bad_input_throws) {
    Market m = kRef;
    m.T = 0.0;
    m.S = 90.0;
    CHECK_NEAR(optlib::pde_price(OptionType::Put, m), 10.0, 1e-12);
    m = kRef;
    m.sigma = 0.0;
    CHECK_THROWS(optlib::pde_price(OptionType::Call, m));
    CHECK_THROWS(optlib::pde_price(OptionType::Call, kRef, 401, 400));
    CHECK_THROWS(optlib::pde_price(OptionType::Call, kRef, 400, 1));
    CHECK_THROWS(optlib::pde_price(OptionType::Call, kRef, 400, 400, 0.0));
}

TEST_MAIN()
