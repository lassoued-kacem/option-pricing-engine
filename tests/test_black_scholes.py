"""Black-Scholes closed form vs independent references.

Expected values come from (a) quoted published prices, (b) the independent
pure-Python oracle in reference.py, (c) analytic limits. Never from optlib.
"""
import math

import pytest

import optlib
import reference
from cases import GRID, TYPES, case_id, is_call

Call, Put = optlib.OptionType.Call, optlib.OptionType.Put


@pytest.mark.parametrize(
    "params, call, put, tol",
    [
        # README reference case, quoted to 4 dp -> tol = half a unit in the last digit.
        (dict(S=100, K=100, r=0.05, sigma=0.2, T=1), 10.4506, 5.5735, 5e-5),
        # Hull, Options Futures & Other Derivatives, S=42 example, quoted to 2 dp.
        (dict(S=42, K=40, r=0.10, sigma=0.2, T=0.5), 4.76, 0.81, 5e-3),
    ],
)
def test_quoted_reference_values(params, call, put, tol):
    # Pins both the library and the oracle to published numbers.
    assert optlib.bs_price(Call, **params) == pytest.approx(call, abs=tol)
    assert optlib.bs_price(Put, **params) == pytest.approx(put, abs=tol)
    assert reference.bs_price(True, **params) == pytest.approx(call, abs=tol)
    assert reference.bs_price(False, **params) == pytest.approx(put, abs=tol)


@pytest.mark.parametrize("otype", TYPES, ids=["call", "put"])
@pytest.mark.parametrize("m", GRID, ids=case_id)
def test_matches_independent_oracle(otype, m):
    # Same formula, two implementations (C++ erfc vs Python erf): only rounding
    # separates them. 1e-10 relative (floor 1e-12 absolute) leaves room for the
    # ~1e-15 relative erf/erfc difference amplified by cancellation in deep OTM.
    want = reference.bs_price(is_call(otype), **m)
    assert optlib.bs_price(otype, **m) == pytest.approx(want, rel=1e-10, abs=1e-12)


def test_analytic_limits():
    p = dict(K=100.0, r=0.05, T=1.0, q=0.02)
    df_r, df_q = math.exp(-p["r"] * p["T"]), math.exp(-p["q"] * p["T"])
    # T = 0: payoff. Exact.
    assert optlib.bs_price(Call, S=110, K=100, r=0.05, sigma=0.2, T=0) == 10.0
    assert optlib.bs_price(Put, S=110, K=100, r=0.05, sigma=0.2, T=0) == 0.0
    # sigma = 0: discounted forward payoff. Exact up to rounding (1e-12).
    assert optlib.bs_price(Call, S=110, sigma=0.0, **p) == pytest.approx(110 * df_q - 100 * df_r, abs=1e-12)
    # S -> 0: put -> K e^{-rT}; N(-d2) = 1 - O(1e-300) at S = 1e-6. Tol 1e-12.
    assert optlib.bs_price(Put, S=1e-6, sigma=0.2, **p) == pytest.approx(100 * df_r - 1e-6 * df_q, abs=1e-12)
    # sigma -> infinity: call -> S e^{-qT} (all moneyness washed out).
    # At sigma*sqrt(T) = 40, N(d1) - 1 and N(d2) are ~1e-89 -> tol 1e-10.
    assert optlib.bs_price(Call, S=100, sigma=40.0, **p) == pytest.approx(100 * df_q, abs=1e-10)


def test_monotonicity():
    # Model-free shape properties: call decreasing in K, increasing in S and sigma.
    base = dict(S=100, r=0.05, sigma=0.25, T=1.0, q=0.01)
    calls = [optlib.bs_price(Call, K=k, **base) for k in (80, 90, 100, 110, 120)]
    assert all(a > b for a, b in zip(calls, calls[1:]))
    vols = [optlib.bs_price(Call, S=100, K=100, r=0.05, sigma=s, T=1) for s in (0.1, 0.2, 0.3)]
    assert vols[0] < vols[1] < vols[2]
