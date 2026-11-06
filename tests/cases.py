"""Shared parameter grid for the validation tests."""
import itertools
import math

import optlib
import reference

R = 0.05
K = 100.0

# Spot (OTM/ATM/ITM for a call), maturity, vol, dividend yield.
SPOTS = (80.0, 100.0, 120.0)
MATURITIES = (0.25, 1.0, 3.0)
VOLS = (0.1, 0.3)
DIVS = (0.0, 0.03)
TYPES = (optlib.OptionType.Call, optlib.OptionType.Put)

GRID = [
    dict(S=S, K=K, r=R, sigma=sig, T=T, q=q)
    for S, T, sig, q in itertools.product(SPOTS, MATURITIES, VOLS, DIVS)
]


def case_id(m: dict) -> str:
    return f"S{m['S']:g}-T{m['T']:g}-v{m['sigma']:g}-q{m['q']:g}"


def is_call(t) -> bool:
    return t == optlib.OptionType.Call


def itm_probability(m: dict) -> float:
    """Risk-neutral P(S_T > K) = N(d2), computed independently of optlib."""
    sd = m["sigma"] * math.sqrt(m["T"])
    d2 = (math.log(m["S"] / m["K"]) + (m["r"] - m["q"] - 0.5 * m["sigma"] ** 2) * m["T"]) / sd
    return reference.norm_cdf(d2)


MIN_EXPECTED_PATHS = 20


def mc_clt_skip_reason(otype, m: dict, n_paths: int, control_variate: bool):
    """Return a skip reason when a k-SE bound is not meaningful, else None.

    Plain / antithetic: the SE is only informative if enough paths land where
    the payoff is non-zero (P = p for a call, 1 - p for a put).
    Control variate: additionally, the paths on the *other* side of the strike
    are what the linear control cannot absorb; if almost none are sampled the
    reported SE collapses towards 0 while the true error does not, so we need
    enough paths on both sides: min(p, 1 - p).
    """
    p = itm_probability(m)
    p_side = min(p, 1 - p) if control_variate else (p if is_call(otype) else 1 - p)
    expected = n_paths * p_side
    if expected < MIN_EXPECTED_PATHS:
        kind = "both sides of the strike" if control_variate else "the payoff region"
        return f"SE unreliable: only {expected:.2g} expected paths in {kind} (< {MIN_EXPECTED_PATHS})"
    return None
