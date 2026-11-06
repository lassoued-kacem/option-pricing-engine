#!/usr/bin/env python3
"""Benchmark the antithetic Monte Carlo estimator in pure Python, NumPy and C++.

All three implement the same algorithm, an exact GBM terminal draw with
antithetic pairs and the SE taken over pair means, at equal path counts on the
reference case. Each timing is the best of --repeats runs (time.perf_counter).
The script prints a Markdown table; the README table is pasted from this
output verbatim.

Usage: .venv/bin/python scripts/benchmark.py [--paths 100000 1000000] [--repeats 5]
"""
import argparse
import math
import pathlib
import platform
import random
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build" / "python"))
import optlib  # noqa: E402

REF = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, q=0.0)


def mc_pure_python(S, K, r, sigma, T, q, n_paths, seed):
    rng = random.Random(seed)
    drift = (r - q - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)
    df = math.exp(-r * T)
    n_pairs = n_paths // 2
    total = 0.0
    total_sq = 0.0
    for _ in range(n_pairs):
        z = rng.gauss(0.0, 1.0)
        up = S * math.exp(drift + vol * z)
        dn = S * math.exp(drift - vol * z)
        y = 0.5 * df * (max(up - K, 0.0) + max(dn - K, 0.0))
        total += y
        total_sq += y * y
    mean = total / n_pairs
    var = (total_sq - n_pairs * mean * mean) / (n_pairs - 1)
    return mean, math.sqrt(var / n_pairs)


def mc_numpy(S, K, r, sigma, T, q, n_paths, seed):
    rng = np.random.default_rng(seed)
    n_pairs = n_paths // 2
    z = rng.standard_normal(n_pairs)
    drift = (r - q - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)
    up = S * np.exp(drift + vol * z)
    dn = S * np.exp(drift - vol * z)
    y = 0.5 * math.exp(-r * T) * (np.maximum(up - K, 0.0) + np.maximum(dn - K, 0.0))
    return float(y.mean()), float(y.std(ddof=1) / math.sqrt(n_pairs))


def mc_cpp(S, K, r, sigma, T, q, n_paths, seed):
    res = optlib.mc_antithetic(optlib.OptionType.Call, S, K, r, sigma, T, n_paths=n_paths, seed=seed, q=q)
    return res.price, res.std_error


METHODS = [("pure Python", mc_pure_python), ("NumPy", mc_numpy), ("C++ (optlib)", mc_cpp)]


def best_time(fn, n_paths, repeats):
    best, out = math.inf, None
    for i in range(repeats):
        t0 = time.perf_counter()
        out = fn(**REF, n_paths=n_paths, seed=1234 + i)
        best = min(best, time.perf_counter() - t0)
    return best, out


def cpu_name():
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=int, nargs="+", default=[100_000, 1_000_000])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    bs = optlib.bs_price(optlib.OptionType.Call, **REF)
    print(f"Machine: {cpu_name()}, {platform.system()} {platform.machine()}, "
          f"Python {platform.python_version()}, NumPy {np.__version__}; single thread")
    print(f"Reference call (S=K=100, r=5%, sigma=20%, T=1): Black-Scholes = {bs:.4f}")
    print(f"Antithetic MC, best of {args.repeats} runs\n")
    print("| Method | Paths | Price | Std. error | Time (ms) | Speed-up vs pure Python |")
    print("|---|---:|---:|---:|---:|---:|")
    for n in args.paths:
        base = None
        for name, fn in METHODS:
            t, (price, se) = best_time(fn, n, args.repeats)
            base = base or t
            print(f"| {name} | {n:,} | {price:.4f} | {se:.4f} | {t * 1e3:,.1f} | {base / t:,.1f}x |")


if __name__ == "__main__":
    main()
