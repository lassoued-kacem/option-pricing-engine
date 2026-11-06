"""Analytic Greeks vs central finite-difference bumps.

Two references per Greek: bumps of the independent Python oracle (fully
independent of the C++ code) and bumps of optlib.bs_price (checks the Greek
formulas are consistent with the library's own pricer).

Step sizes are 1e-2 of each variable's natural scale in Black-Scholes
(d1, d2 move by O(1) when S changes by S*sigma*sqrt(T), sigma by sigma, r by
sigma/sqrt(T), T by T), so truncation error is comparable across the grid.
Richardson extrapolation of central differences, (4 D(h/2) - D(h)) / 3,
removes the O(h^2) term, leaving O(h^4) ~ 1e-8 relative (times a moneyness
factor polynomial in d) plus rounding O(eps * V / h) ~ 1e-12. Tolerance:
1e-6 relative with a 1e-6 absolute floor.
"""
import math

import pytest

import optlib
import reference
from cases import GRID, TYPES, case_id, is_call

KEYS = ("delta", "gamma", "vega", "theta", "rho")


def _central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def _second(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def _richardson(diff, f, x, h):
    return (4 * diff(f, x, h / 2) - diff(f, x, h)) / 3


def fd_greeks(price, m):
    """Richardson-extrapolated finite differences of price(**params). theta = -dV/dT."""
    def along(key):
        return lambda x: price(**dict(m, **{key: x}))

    sd = m["sigma"] * math.sqrt(m["T"])
    hS, hv, hr, hT = 1e-2 * m["S"] * sd, 1e-2 * m["sigma"], 1e-2 * m["sigma"] / math.sqrt(m["T"]), 1e-2 * m["T"]
    return {
        "delta": _richardson(_central, along("S"), m["S"], hS),
        "gamma": _richardson(_second, along("S"), m["S"], hS),
        "vega": _richardson(_central, along("sigma"), m["sigma"], hv),
        "rho": _richardson(_central, along("r"), m["r"], hr),
        "theta": -_richardson(_central, along("T"), m["T"], hT),
    }


@pytest.mark.parametrize("source", ["oracle", "optlib"])
@pytest.mark.parametrize("otype", TYPES, ids=["call", "put"])
@pytest.mark.parametrize("m", GRID, ids=case_id)
def test_greeks_match_bumps(source, otype, m):
    if source == "oracle":
        price = lambda **p: reference.bs_price(is_call(otype), **p)  # noqa: E731
    else:
        price = lambda **p: optlib.bs_price(otype, **p)  # noqa: E731
    want = fd_greeks(price, m)
    got = optlib.bs_greeks(otype, **m)
    for k in KEYS:
        assert got[k] == pytest.approx(want[k], rel=1e-6, abs=1e-6), k


@pytest.mark.parametrize("m", GRID, ids=case_id)
def test_sign_conventions(m):
    dq = math.exp(-m["q"] * m["T"])
    c = optlib.bs_greeks(optlib.OptionType.Call, **m)
    p = optlib.bs_greeks(optlib.OptionType.Put, **m)
    assert 0 < c["delta"] < dq and -dq < p["delta"] < 0
    assert c["gamma"] > 0 and c["vega"] > 0
    assert c["gamma"] == pytest.approx(p["gamma"], rel=1e-12)
    assert c["rho"] > 0 and p["rho"] < 0
