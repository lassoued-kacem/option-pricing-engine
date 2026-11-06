"""Smoke tests for the pybind11 module: every function is callable, returns the
documented types, and errors surface as ValueError. Numerical validation of
the methods lives in the other test modules."""
import math

import pytest

import optlib
import reference
from optlib import OptionType

REF = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def test_reference_case_through_bindings():
    # Quoted reference values (README), 4 decimals -> tolerance 5e-5.
    assert optlib.bs_price(OptionType.Call, **REF) == pytest.approx(10.4506, abs=5e-5)
    assert optlib.bs_price(OptionType.Put, **REF) == pytest.approx(5.5735, abs=5e-5)


def test_oracle_matches_quoted_reference_values():
    # Pins the independent oracle itself to the quoted 4-dp values (tol 5e-5),
    # so tests that use it as ground truth rest on published numbers.
    assert reference.bs_price(True, **REF) == pytest.approx(10.4506, abs=5e-5)
    assert reference.bs_price(False, **REF) == pytest.approx(5.5735, abs=5e-5)


def test_greeks_returns_dict_of_floats():
    g = optlib.bs_greeks(OptionType.Call, **REF)
    assert set(g) == {"delta", "gamma", "vega", "theta", "rho"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in g.values())


Q_CASE = dict(S=97.0, K=103.0, r=0.07, sigma=0.31, T=0.8, q=0.025)


def test_greek_dict_fields_are_not_mixed_up():
    # Parity identities pin each dict entry to its own Greek, independently of
    # the Greek formulas: delta_c - delta_p = e^{-qT}, rho_c - rho_p = K T e^{-rT},
    # theta_c - theta_p = q S e^{-qT} - r K e^{-rT}, gamma and vega equal.
    # r, q, sigma, T all distinct so swapped fields cannot coincide. Tol 1e-10.
    m = Q_CASE
    c = optlib.bs_greeks(OptionType.Call, **m)
    p = optlib.bs_greeks(OptionType.Put, **m)
    dq, dr = math.exp(-m["q"] * m["T"]), math.exp(-m["r"] * m["T"])
    assert c["delta"] - p["delta"] == pytest.approx(dq, abs=1e-10)
    assert c["rho"] - p["rho"] == pytest.approx(m["K"] * m["T"] * dr, abs=1e-10)
    assert c["theta"] - p["theta"] == pytest.approx(m["q"] * m["S"] * dq - m["r"] * m["K"] * dr, abs=1e-10)
    assert c["gamma"] == pytest.approx(p["gamma"], rel=1e-12)
    assert c["vega"] == pytest.approx(p["vega"], rel=1e-12)
    # Each field is the derivative it claims to be: central bump of the bound
    # price (h = 1e-4; O(h^2) error ~1e-7) -> tol 1e-5.
    # Delta and gamma from central first/second differences in S with
    # h = 1e-2 * S: truncation O(h^2) ~1e-7, rounding eps*V/h^2 ~1e-11 -> tol 1e-5.
    hS = 1e-2 * m["S"]
    up, dn = dict(m, S=m["S"] + hS), dict(m, S=m["S"] - hS)
    v_up, v_mid, v_dn = (optlib.bs_price(OptionType.Call, **x) for x in (up, m, dn))
    assert c["delta"] == pytest.approx((v_up - v_dn) / (2 * hS), abs=1e-5)
    assert c["gamma"] == pytest.approx((v_up - 2 * v_mid + v_dn) / hS**2, abs=1e-5)
    h = 1e-4
    for key, greek in (("sigma", "vega"), ("r", "rho")):
        up, dn = dict(m), dict(m)
        up[key] += h
        dn[key] -= h
        fd = (optlib.bs_price(OptionType.Call, **up) - optlib.bs_price(OptionType.Call, **dn)) / (2 * h)
        assert c[greek] == pytest.approx(fd, abs=1e-5), greek


def test_dividend_yield_reaches_every_method():
    # Expected values from the independent pure-Python oracle (tests/reference.py).
    m = Q_CASE
    for otype, is_call in ((OptionType.Call, True), (OptionType.Put, False)):
        want = reference.bs_price(is_call, **m)
        # Closed form: rounding only.
        assert optlib.bs_price(otype, **m) == pytest.approx(want, rel=1e-10)
        # PDE at 800x800: 1e-4 relative accuracy target (see test_pde).
        got = optlib.pde_price(otype, **m, n_space=800, n_time=800)
        assert got == pytest.approx(want, abs=1e-4 * max(want, 1.0))
        # MC: within 4 SE (two-sided false-alarm probability ~6e-5).
        res = optlib.mc_control_variate(otype, **m, n_paths=200_000, seed=3)
        assert abs(res.price - want) < 4 * res.std_error
    # Swapping r and q anywhere in the bindings would break these:
    # the forward e^{(r-q)T} is not symmetric in (r, q).
    swapped = dict(m, r=m["q"], q=m["r"])
    assert reference.bs_price(True, **swapped) != pytest.approx(reference.bs_price(True, **m), rel=1e-3)


@pytest.mark.parametrize(
    "call",
    [
        lambda: optlib.bs_price(OptionType.Call, 100, 100, 0.05, 0.2, 1.0, 0.03),
        lambda: optlib.mc_antithetic(OptionType.Call, 100, 100, 0.05, 0.2, 1.0, 1000),
        lambda: optlib.pde_price(OptionType.Call, 100, 100, 0.05, 0.2, 1.0, 200, 200),
    ],
    ids=["bs_q", "mc_n_paths", "pde_grid"],
)
def test_optional_arguments_are_keyword_only(call):
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize("fn", [optlib.mc_plain, optlib.mc_antithetic, optlib.mc_control_variate])
def test_mc_result_fields(fn):
    res = fn(OptionType.Call, **REF, n_paths=10_000, seed=1)
    assert isinstance(res, optlib.McResult)
    assert res.n_paths == 10_000
    assert res.std_error > 0
    assert "std_error" in repr(res)


def test_mc_seed_is_respected():
    a = optlib.mc_antithetic(OptionType.Put, **REF, n_paths=1000, seed=5)
    b = optlib.mc_antithetic(OptionType.Put, **REF, n_paths=1000, seed=5)
    c = optlib.mc_antithetic(OptionType.Put, **REF, n_paths=1000, seed=6)
    assert a.price == b.price and a.price != c.price


def test_positional_core_arguments():
    pos = optlib.bs_price(OptionType.Call, 100.0, 100.0, 0.05, 0.2, 1.0)
    assert pos == optlib.bs_price(OptionType.Call, **REF)


@pytest.mark.parametrize(
    "call",
    [
        lambda: optlib.bs_price(OptionType.Call, S=-1, K=100, r=0.05, sigma=0.2, T=1),
        lambda: optlib.bs_greeks(OptionType.Call, S=100, K=100, r=0.05, sigma=0.2, T=0),
        lambda: optlib.mc_antithetic(OptionType.Call, **REF, n_paths=7),
        lambda: optlib.pde_price(OptionType.Call, S=100, K=100, r=0.05, sigma=0.0, T=1),
    ],
)
def test_invalid_input_raises_value_error(call):
    with pytest.raises(ValueError):
        call()
