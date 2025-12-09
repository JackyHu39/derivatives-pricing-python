import numpy as np

def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    n_paths: int,
    seed: int = 0,
):
    """
    GBM model: dS = mu*S*dt + sigma*S*dW

    Exact discretization:
      S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)
      where Z ~ N(0,1)

    Returns:
      times: shape (steps+1,)
      paths: shape (n_paths, steps+1)
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    times = np.linspace(0.0, T, steps + 1)

    paths = np.empty((n_paths, steps + 1), dtype=float)
    paths[:, 0] = S0

    # Pre-generate random shocks
    Z = rng.standard_normal(size=(n_paths, steps))

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    # Build paths iteratively
    for t in range(steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion[:, t])

    return times, paths

def simulate_gbm_terminal(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int = 0,
):
    """
    Simulate ONLY S_T directly (terminal price), using the closed form:
      S_T = S0 * exp((mu - 0.5*sigma^2)T + sigma*sqrt(T)*Z)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return ST
    
