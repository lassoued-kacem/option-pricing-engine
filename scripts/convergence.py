#!/usr/bin/env python3
"""Convergence study: Monte Carlo error vs paths, PDE error vs grid size.

MC: for each N = 2^10 .. 2^20, the root-mean-square error against the
Black-Scholes price over R independent seeds, for all three estimators. A
log-log least-squares fit should give a slope of -1/2.
PDE: absolute error against Black-Scholes for N = Nx = Nt = 50 .. 1600. Crank-
Nicolson is second order, so the fitted slope should be about -2.

Prints the fitted slopes and writes two PNGs (default: docs/figures/).

Usage: .venv/bin/python scripts/convergence.py [--seeds 256] [--out docs/figures] [--no-plots]
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build" / "python"))
import optlib  # noqa: E402

REF = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
CALL = optlib.OptionType.Call

MC_PATHS = [2**k for k in range(10, 21)]
PDE_GRIDS = [50, 100, 200, 400, 800, 1600]
ESTIMATORS = [
    ("Plain", optlib.mc_plain),
    ("Antithetic", optlib.mc_antithetic),
    ("Antithetic + control variate", optlib.mc_control_variate),
]

# Validated categorical palette (dataviz reference palette, light surface).
SURFACE = "#fcfcfb"
TEXT, TEXT_2, GRID, REF_LINE = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]


def fmt_slope(s):
    return f"{s:.3f}".replace("-", "\u2212")


def fit_slope(x, y):
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return slope, intercept


def mc_study(n_seeds):
    bs = optlib.bs_price(CALL, **REF)
    rmse = {}
    for name, fn in ESTIMATORS:
        errs = []
        for i, n in enumerate(MC_PATHS):
            prices = np.array([fn(CALL, **REF, n_paths=n, seed=1_000_003 * i + s).price for s in range(n_seeds)])
            errs.append(float(np.sqrt(np.mean((prices - bs) ** 2))))
        rmse[name] = np.array(errs)
    return rmse


def pde_study():
    bs = optlib.bs_price(CALL, **REF)
    return np.array([abs(optlib.pde_price(CALL, **REF, n_space=n, n_time=n) - bs) for n in PDE_GRIDS])


def style_axes(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REF_LINE)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=12)


def plot_mc(rmse, slopes, out, n_seeds):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150, facecolor=SURFACE)
    x = np.array(MC_PATHS, dtype=float)
    for (name, _), color, marker in zip(ESTIMATORS, SERIES, MARKERS):
        y = rmse[name]
        ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{name}  (fitted slope {fmt_slope(slopes[name])})")
    # Reference 1/sqrt(N) line, anchored 1.5x above the plain estimator's first
    # point so it does not overlap the data.
    anchor = 1.5 * rmse[ESTIMATORS[0][0]][0]
    ax.plot(x, anchor * (x / x[0]) ** -0.5, color=REF_LINE, linewidth=1.5, linestyle="--",
            label="Reference slope −1/2 (shifted up for visibility)")
    style_axes(ax, "Number of paths N", f"RMSE vs Black-Scholes ({n_seeds} seeds)",
               "Monte Carlo error falls like 1/√N")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_pde(errs, slope, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150, facecolor=SURFACE)
    x = np.array(PDE_GRIDS, dtype=float)
    ax.plot(x, errs, color=SERIES[0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=f"Crank-Nicolson, Nx = Nt = N  (fitted slope {fmt_slope(slope)})")
    # Shifted down x0.4 so it does not hide the data it parallels.
    ax.plot(x, 0.4 * errs[0] * (x / x[0]) ** -2.0, color=REF_LINE, linewidth=1.5, linestyle="--",
            label="Reference slope −2 (shifted down for visibility)")
    style_axes(ax, "Grid size N", "|PDE − Black-Scholes|", "PDE error falls like 1/N²")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=256)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    rmse = mc_study(args.seeds)
    mc_slopes = {name: fit_slope(MC_PATHS, rmse[name])[0] for name, _ in ESTIMATORS}
    pde_errs = pde_study()
    pde_slope = fit_slope(PDE_GRIDS, pde_errs)[0]

    print(f"Reference call, Black-Scholes = {optlib.bs_price(CALL, **REF):.6f}")
    print(f"\nMonte Carlo RMSE over {args.seeds} seeds, N = 2^10 .. 2^20 (expected slope -0.5)\n")
    print("| Estimator | RMSE at N=2^10 | RMSE at N=2^20 | Fitted slope |")
    print("|---|---:|---:|---:|")
    for name, _ in ESTIMATORS:
        print(f"| {name} | {rmse[name][0]:.4f} | {rmse[name][-1]:.6f} | {mc_slopes[name]:.3f} |")
    print("\nCrank-Nicolson |error|, Nx = Nt = N (expected slope -2)\n")
    print("| N | " + " | ".join(str(n) for n in PDE_GRIDS) + " |")
    print("|---|" + "---:|" * len(PDE_GRIDS))
    print("| error | " + " | ".join(f"{e:.2e}" for e in pde_errs) + " |")
    print(f"\nPDE fitted slope: {pde_slope:.3f}")

    if not args.no_plots:
        args.out.mkdir(parents=True, exist_ok=True)
        plot_mc(rmse, mc_slopes, args.out / "mc_convergence.png", args.seeds)
        plot_pde(pde_errs, pde_slope, args.out / "pde_convergence.png")
        print(f"\nWrote {args.out / 'mc_convergence.png'} and {args.out / 'pde_convergence.png'}")


if __name__ == "__main__":
    main()
