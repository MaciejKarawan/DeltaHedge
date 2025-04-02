import csv
import numpy as np
import scipy.stats as si
from numpy.random import normal
import matplotlib.pyplot as plt


def BM(n, T=1):
    Z = [normal(0, 1) for _ in range(n+1)]
    points = [0]*(n + 1)
    for i in range(1, n+1):
        points[i] = points[i-1] + np.sqrt(T/n)*Z[i]
    return points


def GBM(n, points, alpha, sigma, S0):
    S = [0]*(n + 1)
    S[0] = S0
    t = [i / n for i in range(n+1)]

    for i in range(1, n+1):
        S[i] = S0 * np.exp(alpha * t[i] + sigma * points[i])

    return S


def bs_price_vanilla(S, K, T, r, sigma, type='call'):
    assert type in ['call', 'put']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    if type == 'call':
        return call_price
    else:
        put_price = call_price + K * np.exp(-r * T) - S
        return put_price


def bs_price_binary(S, K, T, r, sigma, type='call'):
    assert type in ['call', 'put']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = np.exp(-r * T) * si.norm.cdf(d2)
    if type == 'call':
        return call_price
    else:
        put_price = np.exp(-r * T) * (1 - si.norm.cdf(d2))
        return put_price


def get_delta_vanilla(S, K, T, r, sigma, type='call'):
    assert type in ['call', 'put']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = si.norm.cdf(d1)
    if type == 'call':
        return delta
    else:
        return delta - 1


def get_delta_binary(S, K, T, r, sigma, type='call'):
    assert type in ['call', 'put']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta_bin = (np.exp(-r * T) * si.norm.pdf(d2))/(sigma * S * np.sqrt(T))
    if type == 'call':
        return delta_bin
    else:
        return -delta_bin


def get_gamma_vanilla(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    gamma = si.norm.pdf(d1)/(S * sigma * np.sqrt(T))
    return gamma


def get_gamma_binary(S, K, T, r, sigma, type='call'):
    assert type in ['call', 'put']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma_bin = (np.exp(-r * T) * d1 * si.norm.pdf(d2))/(sigma**2 * S**2 * T)
    if type == 'call':
        return -gamma_bin
    else:
        return gamma_bin


def delta_gamma_hedge(S0, K, T, r, sigma, mu, n_hedgepoints, type='call'):
    """Simulates a delta hedging strategy for a short call position."""
    # Generate GBM path
    dt = T/n_hedgepoints
    points = BM(n_hedgepoints)
    alpha = mu - sigma**2/2
    S_t = GBM(n_hedgepoints, points, alpha, sigma, S0)

    # Initial Black-Scholes price
    call_price = bs_price_vanilla(S0, K, T, r, sigma, type)
    delta_vanilla = get_delta_vanilla(S0, K, T, r, sigma, type)
    put_binary = bs_price_binary(S0, K, T, r, sigma, type='put')
    delta_binary = get_delta_binary(S0, K, T, r, sigma, type='put')
    gamma_vanilla = get_gamma_vanilla(S0, K, T, r, sigma)
    gamma_binary = get_gamma_binary(S0, K, T, r, sigma, type='put')

    # Współczynniki
    Delta1 = delta_vanilla - gamma_vanilla/gamma_binary * delta_binary
    Delta2 = gamma_vanilla/gamma_binary

    # Chwila 0
    cash_position = call_price - Delta1 * S0 - Delta2 * put_binary
    stock_position = Delta1
    option_position = Delta2

    for i in range(1, len(S_t)-1):
        tau = T - i * dt  # Time remaining
        new_delta_vanilla = get_delta_vanilla(S_t[i], K, tau, r, sigma, type)
        new_put_binary = bs_price_binary(S_t[i], K, tau, r, sigma, type='put')
        new_delta_binary = get_delta_binary(S_t[i], K, tau, r, sigma, type='put')
        new_gamma_vanilla = get_gamma_vanilla(S_t[i], K, tau, r, sigma)
        new_gamma_binary = get_gamma_binary(S_t[i], K, tau, r, sigma, type='put')

        new_Delta1 = new_delta_vanilla - new_gamma_vanilla / new_gamma_binary * new_delta_binary
        new_Delta2 = new_gamma_vanilla / new_gamma_binary

        cash_position = cash_position * np.exp(r * dt) - (new_Delta1 - stock_position) * S_t[i] - \
                        (new_Delta2 - option_position) * new_put_binary
        stock_position = new_Delta1
        option_position = new_Delta2

        if type == 'call':
            payoff = max(S_t[-1] - K, 0)
        else:
            payoff = max(K - S_t[-1], 0)

        if K - S_t[-1] > 0:
            payoff_bin = 1
        else:
            payoff_bin = 0

        final_pnl = cash_position * np.exp(r * dt) + stock_position * S_t[-1] + option_position * payoff_bin - payoff
        return final_pnl


S0, K, T, r, sigma, mu, n_hedgepoints = 2266.92, 1800, 1, 0.0346, 0.2656, 0.0265, 252

print(delta_gamma_hedge(S0, K, T, r, sigma, mu, n_hedgepoints))


def delta_gamma_sample(S0, K, T, r, sigma, mu, n_hedgepoints, n=10000, type='call'):
    pnl_results = []

    for _ in range(n):
        final_pnl = delta_gamma_hedge(S0, K, T, r, sigma, mu, n_hedgepoints, type)
        pnl_results.append(final_pnl)

    return pnl_results


pnl_results = delta_gamma_sample(S0, K, T, r, sigma, mu, n_hedgepoints)
std = np.std(pnl_results)
bins = np.arange(-400, 400, 20)
plt.hist(pnl_results, bins=bins, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(np.mean(pnl_results), color='darkred', linestyle="--", label=f"Mean P/L: {np.mean(pnl_results):.2f}")
plt.axvline(std+np.mean(pnl_results), color='r', linestyle="--", label=f"Standard deviation P/L: {np.std(pnl_results):.2f}")
plt.axvline(np.median(pnl_results), color='green', linestyle="--", label=f"Median P/L: {np.median(pnl_results):.2f}")
plt.axvline(-std+np.mean(pnl_results), color='r', linestyle="--")
plt.xlabel('Wartości zysku/straty', fontsize=12)
plt.ylabel('Częstotliwość', fontsize=12)
plt.legend()
plt.show()
