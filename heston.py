import numpy as np

def simulate_heston_terminal(
    S0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    steps: int,
    n_paths: int,
    seed: int = 0,
):
    """
    Heston model (risk-neutral):
      dS = (r - q)S dt + sqrt(v) S dW1
      dv = kappa(theta - v) dt + xi sqrt(v) dW2
      corr(dW1, dW2) = rho

    Full truncation Euler:
      v is truncated at 0 inside sqrt to avoid negative variance issues.

    Returns:
      ST array of shape (n_paths,)
    """
    rng = np.random.default_rng(seed)
    dt = T / steps

    S = np.full(n_paths, S0, dtype=float)
    v = np.full(n_paths, v0, dtype=float)

    for _ in range(steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)

        # correlate Z2 with Z1
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * Z2

        v_pos = np.maximum(v, 0.0)

        # variance update
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * W2

        v_pos = np.maximum(v, 0.0)

        # price update (log-Euler style)
        S = S * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * W1)

    return S
