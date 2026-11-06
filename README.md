# Option Pricing Engine

A small C++17 library for pricing European options three ways: the
Black-Scholes closed form, Monte Carlo simulation and a finite-difference PDE
solver. Each numerical method is checked against the closed form.

## Building

Requirements: a C++17 compiler, CMake ≥ 3.18 and Python 3 with pybind11, numpy,
pytest and matplotlib.

```bash
python3 -m venv .venv
.venv/bin/pip install pybind11 numpy pytest matplotlib
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=.venv/bin/python
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or run everything (configure, build, C++ tests, Python tests) with
`./scripts/check.sh < /dev/null`.

## Conventions

All prices assume a flat risk-free rate `r` and a continuous dividend yield `q`.
Under the risk-neutral measure the stock drifts at `r − q`.

| Greek | Meaning | Unit |
|-------|---------|------|
| delta | dV/dS | per 1 unit of spot |
| gamma | d²V/dS² | per unit of spot² |
| vega  | dV/dσ | per 1.00 of volatility (divide by 100 for "per vol point") |
| theta | dV/dt, calendar time | per year (divide by 365 for per day) |
| rho   | dV/dr | per 1.00 of rate (divide by 100 for per 1%) |

## Methods

### 1. Black-Scholes closed form

If the stock follows geometric Brownian motion with constant volatility,
the price of a European option has an exact formula:

```
d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T),   d2 = d1 − σ√T
call = S e^{−qT} N(d1) − K e^{−rT} N(d2)
put  = K e^{−rT} N(−d2) − S e^{−qT} N(−d1)
```

`N` is the standard normal CDF. Plainly put, the call is worth the discounted
expected stock received minus the discounted expected strike paid, where each
leg is weighted by its risk-adjusted probability of finishing in the money.
The Greeks are the exact derivatives of these formulas. Edge cases use
their limits: at `T = 0` the price is the payoff, and at `σ = 0` it is the
discounted payoff on the forward.

This formula is the ground truth that the other methods are tested against.
For the reference case `S=K=100, r=5%, σ=20%, T=1` it gives
call ≈ 10.4506 and put ≈ 5.5735.
