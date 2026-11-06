"""Independent Black-Scholes-Merton oracle used only to produce expected values.

Written from the textbook formulas in pure Python (math.erf), sharing no code
with the C++ library. It is itself pinned to quoted published values
(test_bindings.py::test_oracle_matches_quoted_reference_values), so the C++
results are never compared against themselves.
"""
import math


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(is_call: bool, S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    if T == 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    fwd = S * math.exp(-q * T)  # discounted forward of the stock
    pv_k = K * math.exp(-r * T)
    if sigma == 0.0:
        return max(fwd - pv_k, 0.0) if is_call else max(pv_k - fwd, 0.0)
    sd = sigma * math.sqrt(T)
    d1 = math.log(fwd / pv_k) / sd + 0.5 * sd
    d2 = d1 - sd
    if is_call:
        return fwd * norm_cdf(d1) - pv_k * norm_cdf(d2)
    return pv_k * norm_cdf(-d2) - fwd * norm_cdf(-d1)


def forward_parity(S: float, K: float, r: float, T: float, q: float = 0.0) -> float:
    """Model-free C - P = S e^{-qT} - K e^{-rT}."""
    return S * math.exp(-q * T) - K * math.exp(-r * T)
