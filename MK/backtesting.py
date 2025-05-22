import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from scipy import stats
from .hedging import wallet_delta


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


#### Premium

def compute_TVaR(pnl):
    negative_mask = pnl < 0
    p_loss = np.mean(negative_mask)  # P(PnL < 0)
    if p_loss > 0:
        # Średnia strata, gdy wystąpi strata
        cond_mean = np.mean(pnl[negative_mask])
        # TVaR zgodnie z definicją: dzielimy przez prawdopodobieństwo straty
        tvar = cond_mean / p_loss
    else:
        tvar = 0  # Jeżeli nie wystąpiły straty, przyjmujemy 0

    return tvar


def TVar_volrisk(paths, params, K = 1800, option_type='call', vols = np.arange(0.75, 1.01, 0.01)):
    final_TVars = []
    volatilities = vols * params['volatility']
    n = len(vols)
    for vol in volatilities:
        pnl, _ = wallet_delta(paths,
                              T_years=1,
                              sigma=vol,
                              r=params['r'],
                              K=K,
                              option_type=option_type,
                              n_hedge = 252)

        final_pnl = pnl[:-1]

        final_TVars.append(compute_TVaR(final_pnl))

    return np.sum(np.array(final_TVars) * 1/n)

