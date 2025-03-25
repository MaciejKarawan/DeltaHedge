import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


def verify_gbm_paths(paths, s0, mu, sigma, T):
    """Sprawdza właściwości statystyczne wygenerowanych ścieżek"""
    final_values = paths[:, -1]
    expected_mean = s0 * np.exp(mu * T)
    actual_mean = np.mean(final_values)

    log_returns = np.log(paths[:, 1:] / paths[:, :-1])
    expected_volatility = sigma
    actual_volatility = np.std(log_returns) * np.sqrt(252)  # Annualizacja

    print(f"Oczekiwana średnia końcowa: {expected_mean:.2f}")
    print(f"Faktyczna średnia końcowa: {actual_mean:.2f}")
    print(f"Różnica: {100 * (actual_mean - expected_mean) / expected_mean:.2f}%")
    print(f"Oczekiwana zmienność roczna: {expected_volatility:.4f}")
    print(f"Faktyczna zmienność roczna: {actual_volatility:.4f}")
    print(f"Różnica: {100 * (actual_volatility - expected_volatility) / expected_volatility:.2f}%")



### WALLET