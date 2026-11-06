"""Crank-Nicolson PDE vs the Black-Scholes closed form."""
import math

import numpy as np
import pytest

import optlib
import reference
from cases import GRID, TYPES, case_id, is_call

Call = optlib.OptionType.Call


@pytest.mark.parametrize("otype", TYPES, ids=["call", "put"])
@pytest.mark.parametrize("m", GRID, ids=case_id)
def test_fine_grid_accuracy(otype, m):
    # Accuracy spec: 1e-4 relative, floor 1e-4 absolute for cheap options (the
    # grid half-width scales with sigma*sqrt(T), so a relative spec is
    # comparable across cases). The spec is fixed; the grid is the resolution
    # that meets it over this whole grid. At 800 x 800 the worst case
    # (S=K=100, T=3, sigma=0.1 put) has relative error 1.07e-4, just outside
    # the spec, so the sweep runs at 1600 x 1600 (error 4x smaller, O(dx^2)).
    want = reference.bs_price(is_call(otype), **m)
    got = optlib.pde_price(otype, **m, n_space=1600, n_time=1600)
    assert got == pytest.approx(want, abs=1e-4 * max(want, 1.0))


def test_default_grid_reference_case():
    # Plan target: 1e-3 absolute (0.01% of price) at the default 400 x 400 grid.
    m = dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
    assert optlib.pde_price(Call, **m) == pytest.approx(reference.bs_price(True, **m), abs=1e-3)


@pytest.mark.parametrize("m", GRID[::5], ids=case_id)
def test_second_order_in_grid(m):
    # CN + Rannacher is O(dx^2 + dt^2): with Nt = Nx the log-log slope of
    # |error| vs N must be -2. Band [-2.2, -1.8] allows for residual
    # higher-order terms over N = 100..800.
    want = reference.bs_price(True, **m)
    ns = np.array([100, 200, 400, 800])
    errs = np.array([abs(optlib.pde_price(Call, **m, n_space=int(n), n_time=int(n)) - want) for n in ns])
    slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]
    assert -2.2 < slope < -1.8, slope
