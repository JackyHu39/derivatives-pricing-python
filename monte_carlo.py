import numpy as np

def mc_price_european_call_gbm(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    seed: int = 0,
    antithetic: bool = True,
):
    """
    Risk-neutral GBM:
      dS = (r - q) S dt + sigma S dW

    Call Price = exp(-rT) * E[max(S_T - K, 0)]
    Returns: price, standard_error, (low, high) 95% CI
    """
    rng = np.random.default_rng(seed)
    mu_rn = (r - q)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal(half)
        Z = np.concatenate([Z, -Z])
        if len(Z) < n_paths:
            Z = np.concatenate([Z, rng.standard_normal(1)])
    else:
        Z = rng.standard_normal(n_paths)

    ST = S0 * np.exp((mu_rn - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0)
    disc_payoff = np.exp(-r * T) * payoff

    price = float(np.mean(disc_payoff))
    se = float(np.std(disc_payoff, ddof=1) / np.sqrt(len(disc_payoff)))

    lo = price - 1.96 * se
    hi = price + 1.96 * se
    return price, se, (lo, hi)

def mc_price_european_put_gbm(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    seed: int = 0,
    antithetic: bool = True,
):
    """
    Put Price = exp(-rT) * E[max(K - S_T, 0)]
    Returns: price, standard_error, (low, high) 95% CI
    """
    rng = np.random.default_rng(seed)
    mu_rn = (r - q)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal(half)
        Z = np.concatenate([Z, -Z])
        if len(Z) < n_paths:
            Z = np.concatenate([Z, rng.standard_normal(1)])
    else:
        Z = rng.standard_normal(n_paths)

    ST = S0 * np.exp((mu_rn - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(K - ST, 0.0)
    disc_payoff = np.exp(-r * T) * payoff

    price = float(np.mean(disc_payoff))
    se = float(np.std(disc_payoff, ddof=1) / np.sqrt(len(disc_payoff)))

    lo = price - 1.96 * se
    hi = price + 1.96 * se
    return price, se, (lo, hi)
