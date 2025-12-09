from math import log, sqrt, exp
from dataclasses import dataclass
from scipy.stats import norm

@dataclass
class BSInputs:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0

def _d1_d2(x: BSInputs):
    if x.T <= 0 or x.sigma <= 0 or x.S <= 0 or x.K <= 0:
        return float("nan"), float("nan")
    d1 = (log(x.S / x.K) + (x.r - x.q + 0.5 * x.sigma**2) * x.T) / (x.sigma * sqrt(x.T))
    d2 = d1 - x.sigma * sqrt(x.T)
    return d1, d2

def bs_call_price(x: BSInputs) -> float:
    if x.T <= 0:
        return max(x.S - x.K, 0.0)
    d1, d2 = _d1_d2(x)
    return x.S * exp(-x.q * x.T) * norm.cdf(d1) - x.K * exp(-x.r * x.T) * norm.cdf(d2)

def bs_put_price(x: BSInputs) -> float:
    if x.T <= 0:
        return max(x.K - x.S, 0.0)
    d1, d2 = _d1_d2(x)
    return x.K * exp(-x.r * x.T) * norm.cdf(-d2) - x.S * exp(-x.q * x.T) * norm.cdf(-d1)

def bs_delta_call(x: BSInputs) -> float:
    d1, _ = _d1_d2(x)
    return exp(-x.q * x.T) * norm.cdf(d1)

def bs_delta_put(x: BSInputs) -> float:
    d1, _ = _d1_d2(x)
    return exp(-x.q * x.T) * (norm.cdf(d1) - 1.0)

def bs_gamma(x: BSInputs) -> float:
    d1, _ = _d1_d2(x)
    return exp(-x.q * x.T) * norm.pdf(d1) / (x.S * x.sigma * sqrt(x.T))

def bs_vega(x: BSInputs) -> float:
    d1, _ = _d1_d2(x)
    return x.S * exp(-x.q * x.T) * norm.pdf(d1) * sqrt(x.T)

def bs_theta_call(x: BSInputs) -> float:
    d1, d2 = _d1_d2(x)
    term1 = -(x.S * norm.pdf(d1) * x.sigma * exp(-x.q*x.T)) / (2 * sqrt(x.T))
    term2 = x.q * x.S * exp(-x.q*x.T) * norm.cdf(d1)
    term3 = -x.r * x.K * exp(-x.r*x.T) * norm.cdf(d2)
    return term1 + term2 + term3

def bs_theta_put(x: BSInputs) -> float:
    d1, d2 = _d1_d2(x)
    term1 = -(x.S * norm.pdf(d1) * x.sigma * exp(-x.q*x.T)) / (2 * sqrt(x.T))
    term2 = -x.q * x.S * exp(-x.q*x.T) * norm.cdf(-d1)
    term3 = x.r * x.K * exp(-x.r*x.T) * norm.cdf(-d2)
    return term1 + term2 + term3
