**Stochastic Pricing Lab (GBM • Black–Scholes • Monte Carlo • Heston)**

This project is a small “pricing lab” in Python to:

    simulate stock price paths

    price European options

    compute Greeks (and validate them numerically)

    compare GBM (constant volatility) vs Heston (stochastic volatility) tail risk

**Models covered**

GBM (Geometric Brownian Motion): stock paths + terminal distribution

Black-Scholes: closed-form call/put pricing + Greeks

Monte Carlo: pricing under GBM + convergence diagnostics

Heston (optional): stochastic volatility simulation + tail/VaR comparison vs GBM

**What you get (outputs)**

Running run_all.py generates:

    GBM sample paths plot

    GBM terminal price histogram

    Monte Carlo convergence plot vs Black-Scholes (with 95% CI)

    Heston vs GBM terminal log-return distribution plot (fat tails)

Figures are saved to ./figures/ :

    gbm_paths.png

    gbm_terminal_prices.png

    mc_convergence.png

    heston_vs_gbm_tails.png

**Project structure**

stochastic-pricing-lab/
src/
init.py
gbm.py
black_scholes.py
monte_carlo.py
heston.py
figures/ (auto-generated .png outputs)
run_all.py
requirements.txt
README.md

**What each section does**

1. GBM demo

    simulates many 1-year GBM paths

    plots paths and histogram of terminal prices S_T

    prints probability of ending below S0 and percentile stats

2. Black-Scholes + Greeks

    closed-form call/put pricing

    Greeks: Delta, Gamma, Vega

    finite-difference checks (sanity check that Greeks are correct)

3. Monte Carlo pricing convergence

    prices the same BS call via Monte Carlo under GBM (risk-neutral)

    prints MC estimate + 95% CI for increasing path counts

    plots convergence toward BS closed-form

4. Heston vs GBM tails

    compares terminal return distributions under:
        
        a) constant-vol GBM
    
        b) Heston stochastic volatility (fat tails + equity skew)

    prints quantiles, kurtosis, and 1% VaR for both models

    shows why constant volatility can understate tail risk

