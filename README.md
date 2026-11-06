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
`./scripts/check.sh < /dev/null`. To skip the Python module, configure with
`-DOPTLIB_BUILD_PYTHON=OFF`.

## Python usage

The build places the module in `build/python/`:

```python
import sys; sys.path.insert(0, "build/python")
import optlib
from optlib import OptionType

optlib.bs_price(OptionType.Call, S=100, K=100, r=0.05, sigma=0.2, T=1)        # closed form
optlib.bs_greeks(OptionType.Put, S=100, K=100, r=0.05, sigma=0.2, T=1, q=0.02) # dict of Greeks
res = optlib.mc_control_variate(OptionType.Call, S=100, K=100, r=0.05, sigma=0.2, T=1,
                                n_paths=1_000_000, seed=7)
res.price, res.std_error                                                       # estimate and its SE
optlib.pde_price(OptionType.Call, S=100, K=100, r=0.05, sigma=0.2, T=1, n_space=400, n_time=400)
```

`option_type, S, K, r, sigma, T` may be given positionally. Everything after
them is keyword-only: the dividend yield `q` (default 0), `n_paths`/`seed` for
Monte Carlo, and `n_space`/`n_time`/`width_sd` for the PDE. The Monte Carlo
functions are `mc_plain`, `mc_antithetic` and `mc_control_variate`.
Out-of-range values (negative spot or volatility, odd antithetic path counts,
etc.) raise `ValueError`. Arguments of the wrong type, including negative
integers for `n_paths` or `seed`, raise `TypeError`.

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

**Limitation:** the reported SE is only trustworthy when enough paths land on
both sides of the strike. For a very deep in- or out-of-the-money option,
nearly every path gives a payoff that is a straight line in `S_T`. The control
variate absorbs that line exactly, so the SE can collapse towards zero even
though the rarely-sampled region still carries real error.

Every estimator is tested against the Black-Scholes price within a stated
number of standard errors: 4 in the C++ unit tests, 4.5 across the larger
pytest grid. The tests also check that the reported SE matches the actual
scatter of prices across many independent seeds. See [Testing](#testing).

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


## Benchmark: pure Python vs NumPy vs C++

`scripts/benchmark.py` runs the same antithetic Monte Carlo estimator three
ways: a pure-Python loop, vectorised NumPy, and the C++ library through the
Python bindings. All three use equal path counts on the reference call. The
output below is pasted verbatim from one run of
`.venv/bin/python scripts/benchmark.py` (Release build, GCC 16):

```
Machine: AMD Ryzen 7 4800HS with Radeon Graphics, Linux x86_64, Python 3.14.10, NumPy 2.5.4; single thread
Reference call (S=K=100, r=5%, sigma=20%, T=1): Black-Scholes = 10.4506
Antithetic MC, best of 5 runs

| Method | Paths | Price | Std. error | Time (ms) | Speed-up vs pure Python |
|---|---:|---:|---:|---:|---:|
| pure Python | 100,000 | 10.4571 | 0.0329 | 58.1 | 1.0x |
| NumPy | 100,000 | 10.4714 | 0.0329 | 4.4 | 13.2x |
| C++ (optlib) | 100,000 | 10.4361 | 0.0328 | 4.0 | 14.6x |
| pure Python | 1,000,000 | 10.4553 | 0.0104 | 579.1 | 1.0x |
| NumPy | 1,000,000 | 10.4562 | 0.0104 | 48.2 | 12.0x |
| C++ (optlib) | 1,000,000 | 10.4521 | 0.0104 | 39.5 | 14.7x |
```

All three agree with Black-Scholes within about one standard error, as they
should. C++ is about 15× faster than the pure-Python loop but only about 1.2×
faster than NumPy. Both compiled versions spend their time drawing normal
random numbers and computing `exp`. A likely factor is that NumPy's generator
uses a fast ziggurat sampler, while the C++ code uses
`std::normal_distribution`. Timings depend on the machine; rerun the script
to get your own.

## Testing

`./scripts/check.sh < /dev/null` builds everything and runs the C++ unit tests
(CTest) and the pytest suite in `tests/`. Expected values are never taken from
the library's own output. They come from published prices, put-call parity,
analytic limits, and an independent pure-Python Black-Scholes in
`tests/reference.py`, which is itself pinned to the published values. Every
tolerance is stated and justified next to its test:

| Check | Tolerance | Why |
|-------|-----------|-----|
| Closed form vs oracle (36 cases × call/put) | 1e-10 relative | Same formula in two implementations; only rounding differs |
| Put-call parity, closed form | 1e-12 · S | Model-free identity, rounding only |
| Monte Carlo vs closed form | 4.5 standard errors | Keeps the chance of any false alarm across 216 checks below 0.15% |
| MC reported SE vs actual scatter over 300 seeds | ratio in [0.84, 1.16] | ~4σ band for a sample standard deviation |
| PDE vs closed form, 1600×1600 | 1e-4 relative | Accuracy spec; the error shrinks like 1/N² |
| PDE convergence slope | fitted slope in [−2.2, −1.8] | Crank-Nicolson is second order |
| Greeks vs Richardson finite differences | 1e-6 relative | Truncation O(h⁴) plus rounding; worst observed 2.5e-9 |

Some Monte Carlo cases are skipped, with the reason reported. Plain and
antithetic cases are skipped when fewer than 20 simulated paths are expected
to end with a non-zero payoff. Control-variate cases are skipped when fewer
than 20 are expected on either side of the strike. In those cases the barely
sampled region makes the reported standard error unreliable (see the
limitation above), so an "within k standard errors" check means nothing.
