"""Put-call parity C - P = S e^{-qT} - K e^{-rT} for all three methods.

Parity is model-free, so the right-hand side is an independent expectation.
"""
import math

import pytest

import optlib
import reference
from cases import GRID, case_id, mc_clt_skip_reason

Call, Put = optlib.OptionType.Call, optlib.OptionType.Put
N_PATHS = 100_000


@pytest.mark.parametrize("m", GRID, ids=case_id)
def test_parity_black_scholes(m):
    rhs = reference.forward_parity(m["S"], m["K"], m["r"], m["T"], m["q"])
    # Rounding only: 1e-12 relative to spot.
    assert optlib.bs_price(Call, **m) - optlib.bs_price(Put, **m) == pytest.approx(rhs, abs=1e-12 * m["S"])


@pytest.mark.parametrize("m", GRID, ids=case_id)
def test_parity_pde(m):
    rhs = reference.forward_parity(m["S"], m["K"], m["r"], m["T"], m["q"])
    c = optlib.pde_price(Call, **m, n_space=1600, n_time=1600)
    p = optlib.pde_price(Put, **m, n_space=1600, n_time=1600)
    # Discretisation error only: sum of the per-price accuracy target used in
    # test_pde.py (1e-4 relative, floor 1e-4 absolute, met at 1600 x 1600).
    assert c - p == pytest.approx(rhs, abs=1e-4 * (max(c, 1.0) + max(p, 1.0)))


@pytest.mark.parametrize("fn", [optlib.mc_antithetic, optlib.mc_control_variate], ids=["anti", "cv"])
@pytest.mark.parametrize("m", GRID[::3], ids=case_id)
def test_parity_monte_carlo(fn, m):
    # Both legs need a meaningful SE (same rule as test_monte_carlo.py).
    for otype in (Call, Put):
        reason = mc_clt_skip_reason(otype, m, N_PATHS, control_variate=(fn is optlib.mc_control_variate))
        if reason:
            pytest.skip(f"{otype.name}: {reason}")
    rhs = reference.forward_parity(m["S"], m["K"], m["r"], m["T"], m["q"])
    c = fn(Call, **m, n_paths=N_PATHS, seed=11)
    p = fn(Put, **m, n_paths=N_PATHS, seed=12)
    # Independent seeds: SE(C - P) = sqrt(se_c^2 + se_p^2). 4.5 SE: two-sided
    # p ~ 7e-6 per check, so the whole family (24 checks) has a false-alarm
    # probability below 2e-4 (Bonferroni). Plus a 1e-10*S rounding floor.
    se = math.hypot(c.std_error, p.std_error)
    assert c.price - p.price == pytest.approx(rhs, abs=4.5 * se + 1e-10 * m["S"])
