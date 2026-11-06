"""Fast numerical checks of the convergence rates shown in docs/figures.

The expected slopes come from theory, not from the code: Monte Carlo error is
O(N^-1/2) by the central limit theorem; Crank-Nicolson is second order.
"""
import numpy as np

import optlib
import reference

REF = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
CALL = optlib.OptionType.Call


def test_mc_rmse_slope_is_minus_one_half():
    # RMSE over R = 64 seeds at N = 2^10 .. 2^16. Each RMSE has relative error
    # ~1/sqrt(2R) = 9%. The least-squares slope over 7 points spaced ln 2 apart
    # then has a standard error of about 0.09 / (ln 2 * sqrt(28)) = 0.025, so
    # the band [-0.6, -0.4] is a 4-sigma band.
    bs = reference.bs_price(True, **REF)  # independent oracle
    ns = np.array([2**k for k in range(10, 17)])
    rmse = []
    for i, n in enumerate(ns):
        prices = np.array([optlib.mc_antithetic(CALL, **REF, n_paths=int(n), seed=7919 * i + s).price
                           for s in range(64)])
        rmse.append(np.sqrt(np.mean((prices - bs) ** 2)))
    slope = np.polyfit(np.log(ns), np.log(rmse), 1)[0]
    assert -0.6 < slope < -0.4, slope


def test_pde_error_slope_is_minus_two():
    # Nx = Nt = N, N = 100 .. 1600. Error ratios are ~4.00 per doubling here
    # (see test_pde.py), so the band [-2.2, -1.8] allows for higher-order terms.
    bs = reference.bs_price(True, **REF)  # independent oracle
    ns = np.array([100, 200, 400, 800, 1600])
    errs = [abs(optlib.pde_price(CALL, **REF, n_space=int(n), n_time=int(n)) - bs) for n in ns]
    slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]
    assert -2.2 < slope < -1.8, slope
