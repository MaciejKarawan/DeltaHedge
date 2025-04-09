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


def double_binary_model(S0, K, T, r, sigma, mu, n_hedgepoints, type='call'):
    '''Model dla binarek z strike z chwili 0'''
    A = sigma
    C = r + 0.5 * sigma**2
    phi = (1 + np.sqrt(5)) / 2
    K_call = K * (2 - np.exp(-A * phi - C))
    K_put = K * (2 - np.exp(A / phi - C))

    dt = T / n_hedgepoints
    points = BM(n_hedgepoints)
    alpha = mu - sigma ** 2 / 2
    S_t = GBM(n_hedgepoints, points, alpha, sigma, S0)

    # Initial Black-Scholes prices
    call_price = bs_price_vanilla(S0, K, T, r, sigma, type)
    delta_vanilla = get_delta_vanilla(S0, K, T, r, sigma, type)
    gamma_vanilla = get_gamma_vanilla(S0, K, T, r, sigma)
    put_binary = bs_price_binary(S0, K_put, T, r, sigma, type='put')
    call_binary = bs_price_binary(S0, K_call, T, r, sigma, type='call')
    delta_binary_put = get_delta_binary(S0, K_put, T, r, sigma, type='put')
    delta_binary_call = get_delta_binary(S0, K_call, T, r, sigma, type='call')
    gamma_binary_put = get_gamma_binary(S0, K_put, T, r, sigma, type='put')
    gamma_binary_call = get_gamma_binary(S0, K_call, T, r, sigma, type='call')

    # Współczynniki
    alpha = gamma_vanilla * S0 / (S0 * gamma_binary_call + K * gamma_binary_put)
    beta = K / S0 * alpha
    Delta1 = delta_vanilla - alpha * delta_binary_call - beta * delta_binary_put

    # Chwila 0
    cost = (abs(Delta1) * S0 + abs(alpha) * call_binary + abs(beta) * put_binary) * 0.004
    total_cost = cost
    cash_position = call_price - Delta1 * S0 - alpha * call_binary - beta * put_binary - cost
    stock_position = Delta1
    option_call_position = alpha
    option_put_position = beta

    for i in range(1, len(S_t)-1):
        tau = T - i * dt  # Time remaining
        new_delta_vanilla = get_delta_vanilla(S_t[i], K, tau, r, sigma, type)
        new_gamma_vanilla = get_gamma_vanilla(S_t[i], K, tau, r, sigma)
        new_put_binary = bs_price_binary(S_t[i], K_put, tau, r, sigma, type='put')
        new_call_binary = bs_price_binary(S_t[i], K_call, tau, r, sigma, type='call')
        new_delta_binary_put = get_delta_binary(S_t[i], K_put, tau, r, sigma, type='put')
        new_delta_binary_call = get_delta_binary(S_t[i], K_call, tau, r, sigma, type='call')
        new_gamma_binary_put = get_gamma_binary(S_t[i], K_put, tau, r, sigma, type='put')
        new_gamma_binary_call = get_gamma_binary(S_t[i], K_call, tau, r, sigma, type='call')

        new_alpha = new_gamma_vanilla * S_t[i] / (S_t[i] * new_gamma_binary_call + K * new_gamma_binary_put)
        new_beta = K / S_t[i] * new_alpha
        new_Delta1 = new_delta_vanilla - new_alpha * new_delta_binary_call - new_beta * new_delta_binary_put

        cost = (abs(new_Delta1 - stock_position) * S_t[i] + abs(new_alpha - option_call_position) * new_call_binary +
                abs(new_beta - option_put_position) * new_put_binary) * 0.004
        cash_position = cash_position * np.exp(r * dt) - (new_Delta1 - stock_position) * S_t[i] - \
                        (new_alpha - option_call_position) * new_call_binary - \
                        (new_beta - option_put_position) * new_put_binary

        total_cost += cost
        stock_position = new_Delta1
        option_call_position = new_alpha
        option_put_position = new_beta

        if type == 'call':
            payoff = max(S_t[-1] - K, 0)
        else:
            payoff = max(K - S_t[-1], 0)

        payoff_call_binary = 1 if S_t[-1] - K_call > 0 else 0
        payoff_put_binary = 1 if K_put - S_t[-1] > 0 else 0

        final_pnl = cash_position * np.exp(r * dt) + stock_position * S_t[-1] + \
                    option_call_position * payoff_call_binary + option_put_position * payoff_put_binary - payoff

        return final_pnl, option_call_position, option_put_position, total_cost


S0, K, T, r, sigma, mu, n_hedgepoints = 2266.92, 1700, 1, 0.0346, 0.2656, 0.0265, 252

print(double_binary_model(S0, K, T, r, sigma, mu, n_hedgepoints))
