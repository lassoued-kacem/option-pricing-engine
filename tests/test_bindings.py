"""Smoke tests for the pybind11 module: every function is callable, returns the
documented types, and errors surface as ValueError. Numerical validation of
the methods lives in the other test modules."""
import math

import pytest

import optlib
from optlib import OptionType

REF = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def test_reference_case_through_bindings():
    # Quoted reference values (README), 4 decimals -> tolerance 5e-5.
    assert optlib.bs_price(OptionType.Call, **REF) == pytest.approx(10.4506, abs=5e-5)
    assert optlib.bs_price(OptionType.Put, **REF) == pytest.approx(5.5735, abs=5e-5)


def test_greeks_returns_dict_of_floats():
    g = optlib.bs_greeks(OptionType.Call, **REF)
    assert set(g) == {"delta", "gamma", "vega", "theta", "rho"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in g.values())


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


def test_keyword_arguments_and_dividend():
    with_q = optlib.bs_price(OptionType.Call, S=100, K=100, r=0.05, sigma=0.2, T=1, q=0.03)
    without_q = optlib.bs_price(OptionType.Call, **REF)
    assert with_q < without_q  # dividends lower the forward, hence the call value
    assert optlib.pde_price(OptionType.Call, **REF, n_space=100, n_time=100) > 0


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
