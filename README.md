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

### 2. Monte Carlo simulation

Monte Carlo prices the option the direct way. It simulates many possible stock
prices at maturity, computes the payoff for each, averages them and discounts
the average back to today. A European payoff depends only on the final price,
and under geometric Brownian motion that price has an exact distribution, so
no time stepping is needed:

```
S_T = S · exp((r − q − σ²/2)·T + σ·√T·Z),   Z ~ N(0, 1)
```

Because the answer is an average of random draws, it is uncertain. Every
estimate therefore reports a **standard error** (SE), the typical size of its
random error, which shrinks like `1/√N` in the number of paths `N`.

Three estimators are provided:

- **Plain**: independent draws. This is the baseline.
- **Antithetic variates**: each draw `Z` is paired with its mirror `−Z`. When
  one path ends high, its partner ends low, so the pair average is much less
  noisy than two independent draws. The two halves of a pair are not
  independent, so the SE is computed from the `N/2` pair averages.
- **Control variate** (on top of antithetic): the discounted final stock price
  has a known average, `S·e^{−qT}`. If the simulated stock prices came out a
  bit high, the option payoffs probably did too. The estimator corrects the
  option average by `β` times the observed error in the stock average, where
  `β` is fitted from the same simulations.

The tests require every estimator to land within 4 standard errors of the
Black-Scholes price. They also check that the reported SE matches the actual
scatter of prices across 200 independent seeds.

### 3. Crank-Nicolson finite differences (PDE)

The option price `V(S, t)` satisfies the Black-Scholes partial differential
equation. The solver works in log-price `x = ln S` and time-to-maturity
`τ`, where the equation has constant coefficients:

```
∂V/∂τ = ½σ² ∂²V/∂x² + (r − q − ½σ²) ∂V/∂x − rV,    V(x, 0) = payoff(eˣ)
```

It lays a grid over `x` and steps backwards from the payoff at maturity to
today. At each step it solves a tridiagonal linear system (Thomas algorithm,
`O(N)`).

- **Crank-Nicolson** averages the explicit and implicit schemes, which gives
  second-order accuracy in both space and time. Doubling the grid cuts the
  error by about 4×.
- **Rannacher start**: the payoff has a kink at the strike, and plain CN
  turns it into slowly-decaying wiggles. The first step is therefore taken as
  two small fully-implicit steps, which smooth the kink. The strike sits
  exactly on a grid node.
- **Boundaries**: the grid spans `ln K ± (|ln(S/K)| + 6σ√T)`. At its far ends
  the option behaves linearly: a deep in-the-money call is worth
  `S e^{−qτ} − K e^{−rτ}` and a deep out-of-the-money one is worth 0. Those
  values are imposed there.
- The price at the actual spot is read off with cubic interpolation.

The tests check the error against Black-Scholes (≤ 1e-3 on the reference case
at the default 400×400 grid), the factor-of-4 error reduction per grid
doubling, and the boundary handling on a deliberately narrow grid.
