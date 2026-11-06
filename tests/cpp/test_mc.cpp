// Monte Carlo unit tests. The ground truth is the Black-Scholes closed form;
// MC is only ever compared to it with a tolerance expressed in standard errors.
//
// Tolerance choice: |MC - BS| < 4 SE. For a correct unbiased estimator the
// two-sided probability of exceeding 4 SE is ~6e-5 per check. Seeds are fixed,
// so the tests are deterministic, but the bound was chosen from that
// statistical argument, not tuned to the observed output.
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "optlib/black_scholes.hpp"
#include "optlib/monte_carlo.hpp"
#include "test_main.hpp"

using optlib::Market;
using optlib::McResult;
using optlib::OptionType;

namespace {
const Market kCases[] = {
    {100, 100, 0.05, 0.00, 0.20, 1.0},   // reference case
    {100, 120, 0.03, 0.02, 0.30, 2.0},   // OTM call / ITM put, dividends
    {80, 70, 0.01, 0.04, 0.15, 0.5},     // q > r: drift r - q < 0
    {100, 100, 0.05, 0.00, 0.60, 3.0},   // high vol, long dated
};
constexpr std::size_t kN = 200000;
constexpr double kZ = 4.0;

using Estimator = McResult (*)(OptionType, const Market&, std::size_t, std::uint64_t);
const Estimator kEstimators[] = {optlib::mc_plain, optlib::mc_antithetic,
                                 optlib::mc_control_variate};
}  // namespace

TEST(all_estimators_within_4_se_of_black_scholes) {
    std::uint64_t seed = 12345;
    for (const auto& m : kCases)
        for (OptionType type : {OptionType::Call, OptionType::Put})
            for (Estimator est : kEstimators) {
                const McResult r = est(type, m, kN, seed++);
                const double bs = optlib::bs_price(type, m);
                CHECK(r.std_error > 0.0);
                CHECK(r.n_paths == kN);
                CHECK_NEAR(r.price, bs, kZ * r.std_error);
            }
}

// Variance reduction ordering at equal path count. For a monotone payoff the
// antithetic pair is negatively correlated, so its variance is strictly below
// plain MC; the control variate (corr(payoff, S_T) high) reduces it further.
// We require clear margins (factor 0.9) so noise in the SE estimates cannot
// flip the ordering.
TEST(variance_reduction_ordering) {
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        const Market& m = kCases[0];
        const double se_plain = optlib::mc_plain(type, m, kN, 1).std_error;
        const double se_anti = optlib::mc_antithetic(type, m, kN, 2).std_error;
        const double se_cv = optlib::mc_control_variate(type, m, kN, 3).std_error;
        CHECK(se_anti < 0.9 * se_plain);
        CHECK(se_cv < 0.9 * se_anti);
    }
}

// The reported SE must match the actual spread of prices across independent
// seeds. With R = 200 replications the sample std of the prices has relative
// error ~ 1/sqrt(2R) = 5%, so [0.8, 1.2] is a ~4-sigma band. This is what
// catches an antithetic SE computed over 2n correlated halves (too small by
// a factor that depends on the pair correlation).
TEST(reported_se_matches_empirical_spread) {
    const int R = 200;
    const std::size_t n = 4000;
    for (Estimator est : kEstimators) {
        std::vector<double> prices;
        double mean_se = 0.0;
        for (int i = 0; i < R; ++i) {
            const auto r = est(OptionType::Call, kCases[0], n, 1000 + i);
            prices.push_back(r.price);
            mean_se += r.std_error / R;
        }
        double mu = 0.0;
        for (double p : prices) mu += p / R;
        double var = 0.0;
        for (double p : prices) var += (p - mu) * (p - mu) / (R - 1);
        const double ratio = std::sqrt(var) / mean_se;
        CHECK(ratio > 0.8 && ratio < 1.2);
    }
}

// SE scales as 1/sqrt(N): quadrupling paths halves the SE. SE estimates at
// N >= 5e4 have relative error well under 2%, so tolerance 0.05 on the ratio.
TEST(se_scales_as_inverse_sqrt_n) {
    const auto& m = kCases[0];
    const double se1 = optlib::mc_antithetic(OptionType::Call, m, 50000, 7).std_error;
    const double se4 = optlib::mc_antithetic(OptionType::Call, m, 200000, 8).std_error;
    CHECK_NEAR(se4 / se1, 0.5, 0.05);
}

// Degenerate inputs have exact answers even under simulation.
// sigma = 0: every path ends at the forward, so price = BS limit and SE = 0.
TEST(zero_volatility_is_exact) {
    Market m = kCases[0];
    m.sigma = 0.0;
    for (Estimator est : kEstimators) {
        const auto r = est(OptionType::Call, m, 1000, 1);
        CHECK_NEAR(r.price, m.S - m.K * std::exp(-m.r * m.T), 1e-10);
        CHECK_NEAR(r.std_error, 0.0, 1e-12);
    }
}

TEST(same_seed_is_reproducible) {
    const auto a = optlib::mc_antithetic(OptionType::Put, kCases[1], 10000, 99);
    const auto b = optlib::mc_antithetic(OptionType::Put, kCases[1], 10000, 99);
    CHECK(a.price == b.price && a.std_error == b.std_error);
}

TEST(invalid_path_counts_throw) {
    CHECK_THROWS(optlib::mc_antithetic(OptionType::Call, kCases[0], 1001, 1));
    CHECK_THROWS(optlib::mc_control_variate(OptionType::Call, kCases[0], 2, 1));
    CHECK_THROWS(optlib::mc_plain(OptionType::Call, kCases[0], 0, 1));
}

TEST_MAIN()
