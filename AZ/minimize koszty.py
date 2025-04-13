import scipy.optimize as opt
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

'''
S0, K, T, r, sigma, mu, n_hedgepoints = 2266.92, 1700, 1, 0.0346, 0.2656, 0.0265, 52
tau = T
A = sigma * np.sqrt(tau)
maks = K * np.exp(-(r * tau + 1.5 * A ** 2))
K_call = maks * np.exp(r * tau + A**2 + 0.5 * A * np.sqrt(A**2 + 4))
K_put = maks * np.exp(r * tau + A**2 - 0.5 * A * np.sqrt(A**2 + 4))

call_price = bs_price_vanilla(S0, K, T, r, sigma, type='call')
delta_vanilla = get_delta_vanilla(S0, K, T, r, sigma, type='call')
gamma_vanilla = get_gamma_vanilla(S0, K, T, r, sigma)
put_binary = bs_price_binary(S0, K_put, T, r, sigma, type='put')
call_binary = bs_price_binary(S0, K_call, T, r, sigma, type='call')
delta_binary_put = get_delta_binary(S0, K_put, T, r, sigma, type='put')
delta_binary_call = get_delta_binary(S0, K_call, T, r, sigma, type='call')
gamma_binary_put = get_gamma_binary(S0, K_put, T, r, sigma, type='put')
gamma_binary_call = get_gamma_binary(S0, K_call, T, r, sigma, type='call')
'''

'''
Równanie na gammę portfela:
0 = -gamma_vanilla + alfa * gamma_binary_call + beta * gamma_binary_put 
--> beta = (gamma_vanilla - alfa * gamma_binary_call) / gamma_binary_put

Równanie na deltę portfela:
0 = -delta_vanilla + Delta1 + alfa * delta_binary_call + beta * delta_binary_put
--> Delta1 = delta_vanilla - alfa * delta_binary_call - beta * delta_binary_put

Funkcja kosztów:
costs( alfa ) = 0.004 * (call_price * abs(Delta1) + call_binary * abs(alfa) + put_binary * abs(beta))
'''

'''
Aktualizajca:
wszystko new_...
koszty_czyszczenia = 0.004 * (abs(alfa) * call_binary(K_call, tau) + abs(beta) * put_binary(K_put, tau)) 

beta = (new_gamma_vanilla - alfa * new_gamma_binary_call) / new_gamma_binary_put
new_Delta1 = new_delta_vanilla - alfa * new_delta_binary_call - beta * new_delta_binary_put

Nowe koszty:
costs( alfa ) = 0.004 * (new_call_price * abs(Delta1 - new_Delta1) + 
                call_binary(new_K_call, tau) + abs(beta) * put_binary(new_K_put, tau)) 
'''

'''
def costs(alfa):
    beta = (gamma_vanilla - alfa * gamma_binary_call) / gamma_binary_put
    Delta1 = delta_vanilla - alfa * delta_binary_call - beta * delta_binary_put

    return 0.004 * (call_price * abs(Delta1) + call_binary * abs(alfa) + put_binary * abs(beta))


result = opt.minimize_scalar(costs)
print("Minimum:", result.fun)
print("x value at minimum:", result.x)
'''

'''
alfas = [x for x in range(-100, 100, 1)]
results = [costs(x) for x in alfas]
plt.plot(alfas, results)
plt.show()
'''

'''
alfa = 0
beta = (gamma_vanilla - alfa * gamma_binary_call) / gamma_binary_put
Delta1 = delta_vanilla - alfa * delta_binary_call - beta * delta_binary_put
print(alfa, beta, Delta1)
'''


def double_binary_model(S0, K, T, r, sigma, mu, n_hedgepoints, type='call'):
    tau = T
    A = sigma * np.sqrt(tau)
    maks = K * np.exp(-(r * tau + 1.5 * A ** 2))
    K_call = maks * np.exp(r * tau + A ** 2 + 0.5 * A * np.sqrt(A ** 2 + 4))
    K_put = maks * np.exp(r * tau + A ** 2 - 0.5 * A * np.sqrt(A ** 2 + 4))

    dt = T / n_hedgepoints
    points = BM(n_hedgepoints)
    a = mu - sigma ** 2 / 2
    S_t = GBM(n_hedgepoints, points, a, sigma, S0)

    # Initial Black-Scholes prices
    call_price = bs_price_vanilla(S0, K, tau, r, sigma, type)
    delta_vanilla = get_delta_vanilla(S0, K, tau, r, sigma, type)
    gamma_vanilla = get_gamma_vanilla(S0, K, tau, r, sigma)
    put_binary = bs_price_binary(S0, K_put, tau, r, sigma, type='put')
    call_binary = bs_price_binary(S0, K_call, tau, r, sigma, type='call')
    delta_binary_put = get_delta_binary(S0, K_put, tau, r, sigma, type='put')
    delta_binary_call = get_delta_binary(S0, K_call, tau, r, sigma, type='call')
    gamma_binary_put = get_gamma_binary(S0, K_put, tau, r, sigma, type='put')
    gamma_binary_call = get_gamma_binary(S0, K_call, tau, r, sigma, type='call')

    # Współczynniki
    def initial_costs(alpha):
        beta = (gamma_vanilla - alpha * gamma_binary_call) / gamma_binary_put
        Delta1 = delta_vanilla - alpha * delta_binary_call - beta * delta_binary_put

        return 0.004 * (call_price * abs(Delta1) + call_binary * abs(alpha) + put_binary * abs(beta))

    result = opt.minimize_scalar(initial_costs)
    alpha = result.x
    beta = (gamma_vanilla - alpha * gamma_binary_call) / gamma_binary_put
    Delta1 = delta_vanilla - alpha * delta_binary_call - beta * delta_binary_put

    # Chwila 0
    koszt = result.fun
    total_cost = koszt
    cash_position = call_price - Delta1 * S0 - alpha * call_binary - beta * put_binary - koszt
    stock_position = Delta1
    option_call_position = alpha
    option_put_position = beta

    for i in range(1, len(S_t)-1):
        tau = T - i * dt  # Time remaining

        # Zerowanie binarek
        put_binary = bs_price_binary(S_t[i], K_put, tau, r, sigma, type='put')
        call_binary = bs_price_binary(S_t[i], K_call, tau, r, sigma, type='call')
        koszty_zerowania = 0.004 * (abs(alpha) * call_binary + abs(beta) * put_binary)
        total_cost += koszty_zerowania
        cash_position = cash_position * np.exp(r * dt) + alpha * call_binary + beta * put_binary - koszty_zerowania

        A = sigma * np.sqrt(tau)
        maks = K * np.exp(-(r * tau + 1.5 * A ** 2))
        K_call = maks * np.exp(r * tau + A ** 2 + 0.5 * A * np.sqrt(A ** 2 + 4))
        K_put = maks * np.exp(r * tau + A ** 2 - 0.5 * A * np.sqrt(A ** 2 + 4))

        call_price = bs_price_vanilla(S_t[i], K, tau, r, sigma, type)
        delta_vanilla = get_delta_vanilla(S_t[i], K, tau, r, sigma, type)
        gamma_vanilla = get_gamma_vanilla(S_t[i], K, tau, r, sigma)
        put_binary = bs_price_binary(S_t[i], K_put, tau, r, sigma, type='put')
        call_binary = bs_price_binary(S_t[i], K_call, tau, r, sigma, type='call')
        delta_binary_put = get_delta_binary(S_t[i], K_put, tau, r, sigma, type='put')
        delta_binary_call = get_delta_binary(S_t[i], K_call, tau, r, sigma, type='call')
        gamma_binary_put = get_gamma_binary(S_t[i], K_put, tau, r, sigma, type='put')
        gamma_binary_call = get_gamma_binary(S_t[i], K_call, tau, r, sigma, type='call')

        def costs(alpha):
            Delta1 = stock_position
            beta = (gamma_vanilla - alpha * gamma_binary_call) / gamma_binary_put
            new_Delta1 = delta_vanilla - alpha * delta_binary_call - beta * delta_binary_put

            return 0.004 * (call_price * abs(Delta1 - new_Delta1) + call_binary * abs(alpha) + put_binary * abs(beta))

        result = opt.minimize_scalar(costs)
        alpha = result.x
        beta = (gamma_vanilla - alpha * gamma_binary_call) / gamma_binary_put
        new_Delta1 = delta_vanilla - alpha * delta_binary_call - beta * delta_binary_put
        koszt = result.fun
        total_cost += koszt
        cash_position = cash_position - (new_Delta1 - stock_position) * S_t[i] - \
                            alpha * call_binary - \
                            beta * put_binary - koszt

        stock_position = new_Delta1
        option_call_position = alpha
        option_put_position = beta

    if type == 'call':
        payoff = max(S_t[-1] - K, 0)
    else:
        payoff = max(K - S_t[-1], 0)

    payoff_call_binary = 1 if S_t[-1] - K_call > 0 else 0
    payoff_put_binary = 1 if K_put - S_t[-1] > 0 else 0

    final_pnl = cash_position * np.exp(r * dt) + stock_position * S_t[-1] + \
                option_call_position * payoff_call_binary + option_put_position * payoff_put_binary - payoff

    return final_pnl, total_cost


def double_binary_sample(S0, K, T, r, sigma, mu, n_hedgepoints, n=1000, type='call'):
    pnl_results = []
    koszty = []

    for _ in range(n):
        final_pnl, total_cost = double_binary_model(S0, K, T, r, sigma, mu, n_hedgepoints, type)
        pnl_results.append(final_pnl)
        koszty.append(total_cost)

    return pnl_results, koszty


S0, K, T, r, sigma, mu, n_hedgepoints = 2266.92, 1700, 1, 0.0346, 0.2656, 0.0265, 52

pnl_results, koszty = double_binary_sample(S0, K, T, r, sigma, mu, n_hedgepoints)
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

