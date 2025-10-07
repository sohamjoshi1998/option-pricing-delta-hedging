# monte_carlo_pricer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, log, exp, erf
import time

def norm_cdf(x):
    # math.erf wrapped for arrays and scalars
    if np.isscalar(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    else:
        return 0.5 * (1.0 + np.vectorize(erf)(np.array(x) / sqrt(2.0)))

def bsm_price(S, K, r, sigma, T, option_type='call'):
    if T <= 0:
        return max(S-K,0.0) if option_type=='call' else max(K-S,0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if option_type=='call':
        return S*norm_cdf(d1) - K*exp(-r*T)*norm_cdf(d2)
    else:
        return K*exp(-r*T)*norm_cdf(-d2) - S*norm_cdf(-d1)

def bsm_delta(S, K, r, sigma, T, option_type='call'):
    if T <= 1e-12:
        return 1.0 if (option_type=='call' and S>K) else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    return norm_cdf(d1) if option_type=='call' else norm_cdf(d1)-1.0

def generate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=None):
    rng = np.random.default_rng(seed)
    dt = T/steps
    Z = rng.standard_normal((n_paths, steps))
    increments = (r - 0.5*sigma*sigma)*dt + sigma*sqrt(dt)*Z
    logS = np.concatenate([np.zeros((n_paths,1)), np.cumsum(increments, axis=1)], axis=1)
    return S0 * np.exp(logS)  # shape (n_paths, steps+1)

def mc_price_european(S0, K, r, sigma, T, n_paths, seed=None, antithetic=False, option_type='call'):
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths//2
        z1 = rng.standard_normal(half)
        Z = np.concatenate([z1, -z1])
        if n_paths % 2 == 1:
            Z = np.concatenate([Z, rng.standard_normal(1)])
    else:
        Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5*sigma*sigma)*T + sigma*sqrt(T)*Z)
    payoffs = np.maximum(ST - K, 0.0) if option_type=='call' else np.maximum(K-ST, 0.0)
    price = np.exp(-r*T) * np.mean(payoffs)
    stderr = np.exp(-r*T) * np.std(payoffs, ddof=1) / np.sqrt(n_paths)
    return price, stderr

def mc_price_asian_arith(S0, K, r, sigma, T, steps, n_paths, seed=None):
    S = generate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed)
    avg = S[:,1:].mean(axis=1)
    payoffs = np.maximum(avg - K, 0.0)
    price = np.exp(-r*T) * np.mean(payoffs)
    stderr = np.exp(-r*T) * np.std(payoffs, ddof=1) / np.sqrt(n_paths)
    return price, stderr

def simulate_delta_hedging(S_paths, K, r, sigma, T, option_type='call'):
    n_paths, steps_plus_one = S_paths.shape
    steps = steps_plus_one - 1
    dt = T/steps
    growth = exp(r*dt)
    pnl = np.zeros(n_paths)
    for i in range(n_paths):
        path = S_paths[i]
        option_price = bsm_price(path[0], K, r, sigma, T, option_type)
        delta_old = bsm_delta(path[0], K, r, sigma, T, option_type)
        cash = option_price - delta_old*path[0]
        for j in range(1, steps_plus_one):
            cash = cash * growth
            t = j*dt
            T_rem = max(T-t, 0.0)
            delta_new = bsm_delta(path[j], K, r, sigma, T_rem, option_type)
            cash -= (delta_new - delta_old)*path[j]
            delta_old = delta_new
        final_portfolio = cash + delta_old*path[-1]
        payoff = max(path[-1]-K,0.0) if option_type=='call' else max(K-path[-1],0.0)
        pnl[i] = final_portfolio - payoff
    return pnl

# --- Example run (modify N for accuracy/time) ---
if __name__ == '__main__':
    S0, K, r, sigma, T = 100.0, 100.0, 0.01, 0.2, 1.0
    N_price = 30000
    N_hedge = 4000
    steps_daily = 252

    mc_call_price, mc_call_se = mc_price_european(S0, K, r, sigma, T, N_price, seed=42, antithetic=True)
    bs_call = bsm_price(S0, K, r, sigma, T)
    print("BSM analytic  call:", bs_call)
    print("MC call: {:.6f} ± {:.6f}".format(mc_call_price, mc_call_se))
    print("Relative error (%) = ", abs(mc_call_price-bs_call)/bs_call*100)

    mc_asian, se_asian = mc_price_asian_arith(S0, K, r, sigma, T, steps_daily, n_paths=N_price//4, seed=123)
    print("Asian (arithmetic) MC:", mc_asian, "±", se_asian)

    # hedging experiment
    freqs = [('Yearly',1), ('Monthly',12), ('Weekly',52), ('Daily',252)]
    results=[]
    for name,freq in freqs:
        S_paths = generate_gbm_paths(S0, r, sigma, T, freq, N_hedge, seed=100+freq)
        pnl = simulate_delta_hedging(S_paths, K, r, sigma, T)
        results.append((name, freq, pnl.mean(), pnl.std(ddof=1), np.mean(np.abs(pnl))))
    df = pd.DataFrame(results, columns=['Frequency','Steps','MeanErr','Std','MAE'])
    print(df.to_string(index=False))
    # produce the two simple charts: MAE vs steps and histogram for daily P&L
    plt.plot(df['Steps'], df['MAE'], marker='o'); plt.xlabel('Steps'); plt.ylabel('MAE'); plt.title('MAE vs Rebalancing steps'); plt.grid(True); plt.show()
    plt.hist(pnl, bins=40); plt.title('Hedging P&L (Daily)'); plt.show()
