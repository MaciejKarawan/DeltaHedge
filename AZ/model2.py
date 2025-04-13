import numpy as np
import scipy.stats as si
from numpy.random import normal
import matplotlib.pyplot as plt


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


S0, K, T, r, sigma, mu, n_hedgepoints = 2266.92, 1700, 1, 0.0346, 0.2656, 0.0265, 52

tau = 1
A = sigma * np.sqrt(tau)
#C = r + 0.5 * sigma ** 2
#phi = (1 + np.sqrt(5)) / 2
K_call = 1480 * np.exp(r * tau + A**2 + 0.5 * A * np.sqrt(A**2 + 4))
K_put = 1480 * np.exp(r * tau + A**2 - 0.5 * A * np.sqrt(A**2 + 4))

gamma_vanilla = get_gamma_vanilla(S0, K, T, r, sigma)
gamma_binary_put = get_gamma_binary(S0, K_put, T, r, sigma, type='put')
gamma_binary_call = get_gamma_binary(S0, K_call, T, r, sigma, type='call')

# Współczynniki
alpha = gamma_vanilla * S0 / (S0 * gamma_binary_call + K * gamma_binary_put)
beta = K / S0 * alpha

print(-gamma_vanilla + alpha * gamma_binary_call + beta * gamma_binary_put)


vanilla_vector = np.vectorize(get_gamma_vanilla)
gamma_vector = np.vectorize(get_gamma_binary)

strikes = [x for x in range(600, 3200, 20)]
tau = T
# results = vanilla_vector(strikes, K, T, r, sigma)

'''
plt.plot(strikes, results, color='black', linewidth=2.5)
for i in range(1, 52):
    tau = T - i/52
    results = vanilla_vector(strikes, K, tau, r, sigma)
    plt.plot(strikes, results, color='grey', alpha=0.7)
plt.show()
'''

results = vanilla_vector(strikes, K, tau, r, sigma)
results_call = gamma_vector(strikes, K, T, r, sigma, 'call')
results_put = gamma_vector(strikes, K, T, r, sigma, 'put')


model_call = gamma_vector( strikes, K_call, tau, r, sigma, 'call')
model_put = gamma_vector(strikes, K_put, tau, r, sigma, 'put')

sum_results = [results[i] + 640 * model_put[i] for i in range(44)] + [results[i] + 850 * results_call[i]
                                                                      for i in range(44, len(strikes))]

alpha_S = [results[i] / results_call[i] for i in range(45)] + [0 for _ in range(45, len(strikes))]
beta_S = [0 for _ in range(44)] + [results[i] / results_put[i] for i in range(44, len(strikes))]

print(strikes[np.argmax(results)])
print(strikes[np.argmax(results_call)])
print(strikes[np.argmin(results_call)])
print(alpha_S)


# Wykres gamma vanilla, binary call, binary put dla różnych tau i ustalonych K_call K_put
strikes = [x for x in range(600, 3200, 20)]
tau = T - 30/52
results = vanilla_vector(strikes, K, tau, r, sigma)
model_call = gamma_vector( strikes, K_call, tau, r, sigma, 'call')
model_put = gamma_vector(strikes, K_put, tau, r, sigma, 'put')

plt.plot(strikes, results, color='black')
plt.plot(strikes, [1000 * x for x in model_call], color='red')
plt.plot(strikes, [1000 * x for x in model_put], color='blue')
plt.axhline(0, color='green', linestyle="--")
plt.axvline(1480, color='green', linestyle="--")
plt.show()
'''
plt.plot(strikes, [alpha_S[i] * model_call[i] for i in range(len(strikes))], color='red')
plt.plot(strikes, sum_results, color='green')
plt.axvline(K, color='green', linestyle="--")
plt.axvline(K * np.exp(-r - A**2 - 0.5 * A * np.sqrt(A**2 + 4)), color='orange', linestyle="--")
plt.axvline(K * np.exp(-r - A**2 + 0.5 * A * np.sqrt(A**2 + 4)), color='yellow', linestyle="--")
plt.axhline(0, color='pink', linestyle="--")
plt.show()
'''

#plt.plot(strikes, results_call)
#plt.plot(strikes, results_put)
plt.plot(strikes, [alpha_S[i] * results_call[i] for i in range(len(strikes))], color='red')
plt.plot(strikes, [beta_S[i] * results_put[i] for i in range(len(strikes))], color='blue')
#plt.plot(strikes, [1000 * x for x in model_call], color='red')
#plt.plot(strikes, [1000 * x for x in model_put], color='blue')
plt.plot(strikes, [-x for x in results], color='black')
plt.axvline(1480, color='pink', linestyle="--")
plt.axhline(0, color='green', linestyle="--")
plt.show()

'''
model = [alpha[i] * model_call[i] + beta[i] * model_put[i] for i in range(len(strikes))]
plt.plot(strikes, model, color='green')
plt.plot(strikes, results, color='black')
plt.axhline(0, color='pink', linestyle="--")
plt.show()
'''