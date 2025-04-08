import csv
import numpy as np
import scipy.stats as si
from numpy.random import normal
import matplotlib.pyplot as plt

np.seterr(all='ignore')


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


def get_delta_binary(S, K, T, r, sigma, type):
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


def get_gamma_binary(S, K, T, r, sigma, type):
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
    K2 = 1800
    dt = T/n_hedgepoints
    points = BM(n_hedgepoints)
    alpha = mu - sigma**2/2
    S_t = GBM(n_hedgepoints, points, alpha, sigma, S0)
    C = (K/K2)**(r/sigma**2 + 1/2)

    # Initial Black-Scholes price
    call_price = bs_price_vanilla(S0, K, T, r, sigma, type)
    delta_vanilla = get_delta_vanilla(S0, K, T, r, sigma, type)
    put_binary = bs_price_binary(S0, K2, T, r, sigma, type='put')
    delta_binary = get_delta_binary(S0, K2, T, r, sigma, type='put')
    # gamma_vanilla = get_gamma_vanilla(S0, K, T, r, sigma)
    # gamma_binary = get_gamma_binary(S0, K2, T, r, sigma, type='put')

    # Współczynniki
   # Delta2 = gamma_vanilla / gamma_binary
    A = np.log(S0**2/(K*K2)) * np.log(K/K2) / (2 * sigma**2 * T)
    Delta2 = sigma**2 * K2 / (np.log(S0/K2)/T + r + 0.5 * sigma**2) * np.exp(A) * C
    Delta1 = delta_vanilla - Delta2 * delta_binary

    # Chwila 0
    cost = (abs(Delta1)*S0 + abs(Delta2)*put_binary)*0.004
    total_cost = cost
    cash_position = call_price - Delta1 * S0 - Delta2 * put_binary - cost
    stock_position = Delta1
    option_position = Delta2

    for i in range(1, len(S_t)-1):
        tau = T - i * dt  # Time remaining
        new_delta_vanilla = get_delta_vanilla(S_t[i], K, tau, r, sigma, type)
        new_put_binary = bs_price_binary(S_t[i], K2, tau, r, sigma, type='put')
        new_delta_binary = get_delta_binary(S_t[i], K2, tau, r, sigma, type='put')
        # new_gamma_vanilla = get_gamma_vanilla(S_t[i], K, tau, r, sigma)
        # new_gamma_binary = get_gamma_binary(S_t[i], K2, tau, r, sigma, type='put')

        #new_Delta2 = new_gamma_vanilla / new_gamma_binary
        A = np.log(S_t[i] ** 2 / (K * K2)) * np.log(K / K2) / (2 * sigma ** 2 * tau)
        if A > 10:
            A = 10
        new_Delta2 = sigma ** 2 * K2 / (np.log(S_t[i] / K2) / tau + r + 0.5 * sigma ** 2) * np.exp(A) * C
        new_Delta1 = new_delta_vanilla - new_Delta2 * new_delta_binary

        cost = (abs(new_Delta1 - stock_position)*S_t[i] + abs(new_Delta2 - option_position)*new_put_binary) * 0.004
        cash_position = cash_position * np.exp(r * dt) - (new_Delta1 - stock_position) * S_t[i] - \
                        (new_Delta2 - option_position) * new_put_binary - cost
        total_cost += cost
        # print('zmiana opcji:', abs(new_Delta2 - option_position), 'cena opcji:', new_put_binary, 'koszty:', cost)

        stock_position = new_Delta1
        option_position = new_Delta2

    if type == 'call':
        payoff = max(S_t[-1] - K, 0)
    else:
        payoff = max(K - S_t[-1], 0)

    payoff_bin = 1 if K2 - S_t[-1] > 0 else 0

    final_pnl = cash_position * np.exp(r * dt) + stock_position * S_t[-1] + option_position * payoff_bin - payoff
    return final_pnl, option_position, total_cost


S0, K, T, r, sigma, mu, n_hedgepoints = 2266.92, 1700, 1, 0.0346, 0.2656, 0.0265, 52

print(delta_gamma_hedge(S0, K, T, r, sigma, mu, n_hedgepoints))


def delta_gamma_sample(S0, K, T, r, sigma, mu, n_hedgepoints, n=500, type='call'):
    pnl_results = []
    koszty = []

    for _ in range(n):
        final_pnl, _, total_cost = delta_gamma_hedge(S0, K, T, r, sigma, mu, n_hedgepoints, type)
        pnl_results.append(final_pnl)
        koszty.append(total_cost)

    return pnl_results, koszty


pnl_results, koszty = delta_gamma_sample(S0, K, T, r, sigma, mu, n_hedgepoints)
std = np.std(pnl_results)
#bins = np.arange(-400, 400, 20)
plt.hist(pnl_results, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(np.mean(pnl_results), color='darkred', linestyle="--", label=f"Mean P/L: {np.mean(pnl_results):.2f}")
plt.axvline(std+np.mean(pnl_results), color='r', linestyle="--", label=f"Standard deviation P/L: {np.std(pnl_results):.2f}")
plt.axvline(np.median(pnl_results), color='green', linestyle="--", label=f"Median P/L: {np.median(pnl_results):.2f}")
plt.axvline(-std+np.mean(pnl_results), color='r', linestyle="--")
plt.axvline(np.mean(koszty), color='green', linestyle="--", label=f"Srednie koszty: {np.mean(koszty):.2f}")
plt.axvline(np.median(koszty), color='green', linestyle="--", label=f"Mediana kosztów: {np.median(koszty):.2f}")
plt.xlabel('Wartości zysku/straty', fontsize=12)
plt.ylabel('Częstotliwość', fontsize=12)
plt.legend()
plt.show()

