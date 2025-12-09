import numpy as np
import matplotlib.pyplot as plt

from src.gbm import simulate_gbm_paths, simulate_gbm_terminal
from src.black_scholes import (
    BSInputs, bs_call_price, bs_put_price,
    bs_delta_call, bs_gamma, bs_vega
)
from src.monte_carlo import mc_price_european_call_gbm, mc_price_european_put_gbm

# Optional:
from src.heston import simulate_heston_terminal


import os
os.makedirs("figures", exist_ok=True)

def savefig(fig, name: str):
    fig.savefig(os.path.join("figures", f"{name}.png"), dpi=200, bbox_inches="tight")



def finite_diff_delta(price_fn, x: BSInputs, eps=1e-4):
    up = BSInputs(S=x.S * (1 + eps), K=x.K, T=x.T, r=x.r, sigma=x.sigma, q=x.q)
    dn = BSInputs(S=x.S * (1 - eps), K=x.K, T=x.T, r=x.r, sigma=x.sigma, q=x.q)
    return (price_fn(up) - price_fn(dn)) / (up.S - dn.S)

def finite_diff_gamma(price_fn, x: BSInputs, eps=1e-4):
    up = BSInputs(S=x.S * (1 + eps), K=x.K, T=x.T, r=x.r, sigma=x.sigma, q=x.q)
    mid = x
    dn = BSInputs(S=x.S * (1 - eps), K=x.K, T=x.T, r=x.r, sigma=x.sigma, q=x.q)
    # second derivative approx
    return (price_fn(up) - 2*price_fn(mid) + price_fn(dn)) / ((x.S*eps)**2)

def section_1_gbm_demo():
    print("\n=== 1) GBM path simulation demo ===")
    S0, mu, sigma = 100, 0.08, 0.20
    T, steps, n_paths = 1.0, 252, 50

    t, paths = simulate_gbm_paths(S0, mu, sigma, T, steps, n_paths, seed=1)

    # GBM paths
    fig = plt.figure()
    for i in range(min(n_paths, 50)):
        plt.plot(t, paths[i], linewidth=1)
    plt.title("GBM sample paths (real-world drift mu)")
    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    savefig(fig, "gbm_paths")
    plt.show()
    plt.close(fig)

    # Terminal distribution
    ST = paths[:, -1]
    fig = plt.figure()
    plt.hist(ST, bins=20)
    plt.title("Terminal prices S_T (GBM)")
    plt.xlabel("S_T")
    plt.ylabel("Count")
    savefig(fig, "gbm_terminal_prices")
    plt.show()
    plt.close(fig)

    prob_loss = np.mean(ST < 100)
    p05 = np.percentile(ST, 5)
    p50 = np.percentile(ST, 50)
    p95 = np.percentile(ST, 95)

    print(f"P(S_T < 100) ≈ {prob_loss:.2%}")
    print(f"5th/50th/95th percentiles: {p05:.2f}, {p50:.2f}, {p95:.2f}")


def section_2_bs_prices_and_greeks():
    print("\n=== 2) Black-Scholes price + Greeks + finite-difference validation ===")

    x = BSInputs(S=100, K=100, T=1.0, r=0.04, sigma=0.20, q=0.0)

    call = bs_call_price(x)
    put = bs_put_price(x)
    print(f"BS Call Price: {call:.4f}")
    print(f"BS Put  Price: {put:.4f}")

    # Greeks (analytic)
    d_call = bs_delta_call(x)
    g = bs_gamma(x)
    v = bs_vega(x)
    print(f"Delta (call): {d_call:.4f}")
    print(f"Gamma:        {g:.6f}")
    print(f"Vega:         {v:.4f} (per 1.00 vol)")

    # Greeks (finite difference checks)
    d_fd = finite_diff_delta(bs_call_price, x)
    g_fd = finite_diff_gamma(bs_call_price, x)
    print(f"Delta FD check: {d_fd:.4f}")
    print(f"Gamma FD check: {g_fd:.6f}")

def section_3_mc_convergence():
    print("\n=== 3) Monte Carlo pricing convergence vs Black-Scholes ===")

    # Risk-neutral parameters for pricing
    S0, K, T = 100, 100, 1.0
    r, q, sigma = 0.04, 0.0, 0.20

    x = BSInputs(S=S0, K=K, T=T, r=r, sigma=sigma, q=q)
    bs = bs_call_price(x)

    Ns = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    mc_prices = []
    ci_los = []
    ci_his = []

    for i, N in enumerate(Ns):
        p, se, (lo, hi) = mc_price_european_call_gbm(
            S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
            n_paths=N, seed=42 + i, antithetic=True
        )
        mc_prices.append(p)
        ci_los.append(lo)
        ci_his.append(hi)
        print(f"N={N:>7} | MC={p:.4f} | 95% CI=({lo:.4f}, {hi:.4f})")

    fig = plt.figure()
    plt.plot(Ns, mc_prices, marker="o", label="MC estimate")
    plt.plot(Ns, [bs]*len(Ns), label="BS closed-form")
    plt.fill_between(Ns, ci_los, ci_his, alpha=0.2, label="MC 95% CI")
    plt.xscale("log")
    plt.title("MC convergence to BS price (log x-axis)")
    plt.xlabel("Number of paths (log scale)")
    plt.ylabel("Option price")
    plt.legend()

    savefig(fig, "mc_convergence")
    plt.show()
    plt.close(fig)


def section_4_heston_vs_gbm_tails():
    print("\n=== 4) Optional: Heston vs GBM tails (terminal returns) ===")

    S0 = 100
    T = 1.0
    r, q = 0.04, 0.0

    # Compare GBM with sigma=20%
    sigma_gbm = 0.20
    ST_gbm = simulate_gbm_terminal(S0=S0, mu=(r-q), sigma=sigma_gbm, T=T, n_paths=200_000, seed=1)

    # Heston parameters (reasonable demo values, not calibrated)
    v0 = 0.20**2
    kappa = 2.0     # mean reversion speed
    theta = 0.20**2 # long-run variance
    xi = 0.60       # vol-of-vol (bigger -> fatter tails)
    rho = -0.6      # negative correlation -> equity skew intuition

    ST_h = simulate_heston_terminal(
        S0=S0, v0=v0, r=r, q=q, kappa=kappa, theta=theta, xi=xi, rho=rho,
        T=T, steps=252, n_paths=200_000, seed=2
    )

    # Compare log returns
    lr_gbm = np.log(ST_gbm / S0)
    lr_h = np.log(ST_h / S0)

    def summarize(x, name):
        q01, q05, q50, q95, q99 = np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])
        kurt = np.mean((x - x.mean())**4) / (np.var(x)**2)  # crude kurtosis
        print(f"{name}: q01={q01:.4f}, q05={q05:.4f}, q50={q50:.4f}, q95={q95:.4f}, q99={q99:.4f}, kurt≈{kurt:.2f}")

    summarize(lr_gbm, "GBM log-returns")
    summarize(lr_h,   "Heston log-returns")
    
    var_1_gbm = np.quantile(lr_gbm, 0.01)
    var_1_h   = np.quantile(lr_h,   0.01)
    print("1% VaR (log-return): GBM", var_1_gbm, "Heston", var_1_h)

    def log_to_simple(lr): 
        return np.exp(lr) - 1

    print("1% VaR (simple return): GBM", log_to_simple(var_1_gbm), "Heston", log_to_simple(var_1_h))

    fig = plt.figure()
    plt.hist(lr_gbm, bins=200, density=True, alpha=0.5, label="GBM")
    plt.hist(lr_h,   bins=200, density=True, alpha=0.5, label="Heston")
    plt.xlim(-0.5, 0.5)
    plt.title("Terminal log-return distribution: Heston vs GBM")
    plt.xlabel("log(S_T / S0)")
    plt.ylabel("Density")
    plt.legend()

    savefig(fig, "heston_vs_gbm_tails")
    plt.show()
    plt.close(fig)






    fig = plt.figure()
    plt.hist(lr_gbm, bins=200, density=True, alpha=0.5, label="GBM")
    plt.hist(lr_h, bins=200, density=True, alpha=0.5, label="Heston")
    plt.xlim(-0.5, 0.5)
    plt.title("Terminal log-return distribution: Heston vs GBM")
    plt.xlabel("log(S_T / S0)")
    plt.ylabel("Density")
    plt.legend()
    savefig(fig, "heston_vs_gbm_tails")
    plt.show()
    plt.close(fig)



if __name__ == "__main__":
    section_1_gbm_demo()
    section_2_bs_prices_and_greeks()
    section_3_mc_convergence()
    section_4_heston_vs_gbm_tails()


