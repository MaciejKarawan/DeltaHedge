import csv
import numpy as np
import scipy.stats as si
from numpy.random import normal
import matplotlib.pyplot as plt

dane = []
with open('/Users/aleksanderzurowski/Downloads/Dane historyczne 2022.csv') as csvfile:
    rows = csv.reader(csvfile, delimiter=",")
    for row in rows:
        dane.append(row[1])

dane.remove('Ostatnio')
dane_hist = [float(slowo.replace('.', '').replace(',', '.')) for slowo in dane]
dane_hist.append(2266.92)
dane_hist.reverse()


def bs_price(S, K, T, r, sigma, type='call'):
    assert type in ['call', 'put']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    if type == 'call':
        return call_price
    else:
        put_price = call_price + K * np.exp(-r * T) - S
        return put_price


def get_delta(S, K, T, r, sigma, type='call'):
    assert type in ['call', 'put']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = si.norm.cdf(d1)
    if type == 'call':
        return delta
    else:
        return delta - 1


def delta_hedge_historical(S0, K, T, r, sigma, mu, n_hedgepoints, type='call'):
    assert type in ['call', 'put']
    """Simulates a delta hedging strategy for a short call position."""
    # Generate GBM path
    dt = T/n_hedgepoints
    S_t = [dane_hist[int(252/n_hedgepoints)*i] for i in range(n_hedgepoints)]
    # Initial Black-Scholes price
    call_price = bs_price(S0, K, T, r, sigma, type)
    delta = get_delta(S0, K, T, r, sigma, type)

    # Short the call, buy delta shares
    cash_position = call_price - delta * S0  # Initial cash position
    stock_position = delta
    portfolio_value = call_price
    stan_portfela = [portfolio_value]
    deltas = [delta]
    cashes = [cash_position]
    option_prices = [call_price]

    # Iterate over time steps to rebalance hedge
    for i in range(1, len(S_t)-1):
        tau = T - i * dt  # Time remaining
        new_delta = get_delta(S_t[i], K, tau, r, sigma, type)
        cash_position = cash_position * np.exp(r * dt) - (new_delta - stock_position) * S_t[i]
        stock_position = new_delta
        portfolio_value = cash_position + stock_position * S_t[i]
        stan_portfela.append(portfolio_value)
        deltas.append(new_delta)
        cashes.append(cash_position)
        option_prices.append(bs_price(S_t[i], K, tau, r, sigma, type))

    # Final profit/loss
    if type == 'call':
        payoff = max(S_t[-1] - K, 0)
    else:
        payoff = max(K - S_t[-1], 0)
    final_pnl = cash_position * np.exp(r * dt) + stock_position * S_t[-1] - payoff  # Payoff of the short call
    stan_portfela.append(final_pnl+payoff)
    cashes.append(cash_position * np.exp(r * dt))
    option_prices.append(payoff)
    return final_pnl, S_t, stan_portfela, deltas, cashes, option_prices


def get_final_pnl(K):
    final_pnl, _, _, _, _, _ = delta_hedge_historical(S0, K, T, r, sigma, mu, n_hedgepoints, type)
    return final_pnl


type = 'put'
S0, K, T, r, sigma, mu, n_hedgepoints = 2266.92, 1800, 1, 0.0346, 0.2656, 0.0265, 252
final_pnl, S_t, stan_portfela, deltas, cashes, option_prices = delta_hedge_historical(S0, K, T, r, sigma, mu, n_hedgepoints, type='put')
print('Finalny zysk/strata z hedgowania put:', final_pnl)
print('Cena put w chwili 0:', option_prices[0])

vectorized_func = np.vectorize(get_final_pnl)
strikes = np.array([x for x in range(700, 1780, 20)] + [1785, 1790, 1792, 1795] +
                     [x for x in range(1800, 3200, 20)] + [x for x in range(3200, 4800, 50)])

results = vectorized_func(strikes)
plt.plot(strikes, results, linestyle='-', color='red', label='Put')


type = 'call'
final_pnl2, S_t, stan_portfela2, deltas2, cashes2, option_prices2 = delta_hedge_historical(S0, K, T, r, sigma, mu, n_hedgepoints, type='call')
print('Finalny zysk/strata z hedgowania call:', final_pnl2)
print('Cena call w chwili 0:', option_prices2[0])
results = vectorized_func(strikes)
plt.plot(strikes, results, linestyle='-', color='blue', label='Call')
plt.axvline(1792, color='black', linestyle="--", label='Wartość S_T')
plt.axhline(0, color='black', linestyle="--")
plt.xlabel('Strike', fontsize=12)
plt.ylabel('Zysk/strata', fontsize=12)
plt.title('Zysk/strata z delta hedgingu w zależności od wartości strike opcji', fontsize=14, fontweight="bold")
plt.legend()
plt.show()

#wykresy dla put
time = [t/(n_hedgepoints-1) for t in range(n_hedgepoints)]
deltas.append(deltas[-1])
plt.plot(time, [x for x in cashes], label='Cash')
plt.plot(time, [1000*d for d in deltas], label='1000*delta akcji')
plt.axhline(0, color='black', linestyle="--")
#plt.ylim(-1550, 1550)
plt.title('Skład portfela w czasie', fontsize=14, fontweight="bold")
plt.legend()
plt.show()

plt.plot(time, stan_portfela, label='Wartość portfela')
plt.plot(time, option_prices, label='Wartość opcji')
plt.axhline(0, color='black', alpha=0.7, linestyle="--")
plt.legend()
plt.show()

narastajace = [stan_portfela[i] - option_prices[i] for i in range(n_hedgepoints)]
narastajace2 = [stan_portfela2[i] - option_prices2[i] for i in range(n_hedgepoints)]
plt.plot(time, narastajace, color='red', label='Put')
plt.plot(time, narastajace2, color='blue', label='Call')
plt.title('Narastający zysk/strata w czasie (wartość portfela - teoretyczna cena opcji w chwili t)')
plt.legend()
plt.show()

payoff = [K - S for S in S_t]
plt.plot(time, payoff, label='Wartość aktywa przesunięta o strike')
plt.plot(time, stan_portfela, label='Wartość portfela')
plt.axhline(0, color='black', alpha=0.7, linestyle="--")
plt.legend()
plt.show()
