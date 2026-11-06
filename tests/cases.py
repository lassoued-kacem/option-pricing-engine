"""Shared parameter grid for the validation tests."""
import itertools

import optlib

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
