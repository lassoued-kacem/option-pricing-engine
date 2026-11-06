"""Monte Carlo vs the Black-Scholes closed form, across the parameter grid.

Tolerance: |MC - BS| < 4.5 SE. Per check the two-sided false-alarm
probability is ~7e-6; the family below has 3 x 36 x 2 = 216 checks, so the
probability that any fails for a correct estimator is < 1.5e-3 (Bonferroni).
Seeds are fixed (deterministic) and were not chosen by looking at results.
"""
import math

import pytest

import optlib
import reference
from cases import GRID, TYPES, case_id, is_call

N_PATHS = 50_000
Z = 4.5
ESTIMATORS = {
    "plain": optlib.mc_plain,
    "antithetic": optlib.mc_antithetic,
    "control_variate": optlib.mc_control_variate,
}


def itm_probability(m):
    """Risk-neutral P(S_T > K) = N(d2), computed independently of optlib."""
    sd = m["sigma"] * math.sqrt(m["T"])
    d2 = (math.log(m["S"] / m["K"]) + (m["r"] - m["q"] - 0.5 * m["sigma"] ** 2) * m["T"]) / sd
    return reference.norm_cdf(d2)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("otype", TYPES, ids=["call", "put"])
@pytest.mark.parametrize("idx, m", list(enumerate(GRID)), ids=[case_id(m) for m in GRID])
def test_within_z_standard_errors(name, otype, idx, m):
    # A k-SE bound relies on the CLT. If almost no path finishes on one side of
    # the strike (deep OTM: payoff ~always 0; deep ITM: payoff ~always linear,
    # so the control variate absorbs it exactly), the sample SE is ~0 and the
    # estimate is dominated by a handful of rare paths. We require at least 20
    # expected paths on each side, using the risk-neutral P(S_T > K) = N(d2).
    p_itm = itm_probability(m)
    if N_PATHS * min(p_itm, 1 - p_itm) < 20:
        pytest.skip(f"CLT not applicable: N*min(p,1-p) = {N_PATHS * min(p_itm, 1 - p_itm):.2g} < 20")
    res = ESTIMATORS[name](otype, **m, n_paths=N_PATHS, seed=1000 + idx)
    want = reference.bs_price(is_call(otype), **m)
    assert res.n_paths == N_PATHS
    assert math.isfinite(res.std_error) and res.std_error > 0
    assert abs(res.price - want) < Z * res.std_error, (res, want)


def test_standard_error_reported_matches_scatter():
    # Reported SE vs actual spread over R independent runs. The sample std of
    # R = 300 prices has relative error ~1/sqrt(2R) = 4%, so [0.84, 1.16] is a
    # ~4-sigma band. Catches an SE that ignores antithetic pairing.
    R = 300
    m = dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0, q=0.0)
    for fn in ESTIMATORS.values():
        runs = [fn(optlib.OptionType.Call, **m, n_paths=2000, seed=s) for s in range(R)]
        prices = [r.price for r in runs]
        mean = sum(prices) / R
        spread = math.sqrt(sum((p - mean) ** 2 for p in prices) / (R - 1))
        mean_se = sum(r.std_error for r in runs) / R
        assert 0.84 < spread / mean_se < 1.16
