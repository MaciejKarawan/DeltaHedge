import numpy as np
import pandas as pd
from .pricing import option_delta, option_price
from scipy.stats import norm

def wallet_delta(paths, T_years, sigma, r, K, option_type='call', n_hedge=252):
    """
    Implementuje portfel hedgujący dla długiej pozycji w opcji, short akcji i inwestycji wolnej od ryzyka.
    Zakładamy że w momencie zero zaczynamy trading. Wykonanie akcji odbywa się w ostatnim kroku.
    (ostatnim dniu, bądź na zamknięcie giełdy ostatniego dnia)

    Parameters:
    -----------
    paths: ndarray
        Macierz ścieżek ceny aktywa kształtu (N, steps), gdzie N to liczba ścieżek,
        a steps to liczba punktów czasowych (włączając t=0)
    T_years: float
        Liczba lat do wygaśnięcia opcji
    sigma: float
        Zmienność roczna
    r: float
        Stopa procentowa wolna od ryzyka (ciągła, roczna)
    K: float
        Cena wykonania opcji (strike)
    option_type: str
        Typ opcji: 'call' lub 'put'
    n_hedge: int
        Liczba rehedgingów w ciągu roku.

    Returns:
    --------
    pnl: ndarray
        Wektor P&L w momencie wykonania opcji
    portfolio_composition: dict
        Słownik zawierający skład portfela w czasie:
        - 'stock': pozycja w indeksie
        - 'cash': gotówka (instrument wolny od ryzyka)
        - 'option_value': teoretyczna wartość opcji
        - 'portfolio_value': całkowita wartość portfela
    """

    N, steps = paths.shape  # N — liczba ścieżek, steps — liczba punktów czasowych
    dt = T_years / (steps) # Obliczanie kroku czasowego (dt)

    if n_hedge > (steps): raise ValueError('Paths should be longer')

    hedge_frequency = (steps) / n_hedge # Obliczanie co ile kroków zmiany aktywa wykonujemy rehedging
    hedge_frequency = max(1, hedge_frequency)  # Minimalna wartość to 1
    hedge_indices = np.arange(0, steps, hedge_frequency) # Tworzenie indeksów dla rehedgingów

    # Inicjalizacja output
    stock_position = np.zeros((N, steps))
    cash_position = np.zeros((N, steps))
    option_value = np.zeros((N, steps))
    portfolio_value = np.zeros((N, steps))

    # Inicjalizacja portfela
    S0 = paths[:, 0]
    t_to_maturity = T_years
    option_value[:, 0] = option_price(S0, K, t_to_maturity, r, sigma, option_type)
    delta = option_delta(S0, K, t_to_maturity, r, sigma, option_type)

    # Początkowy skład portfela
    stock_position[:, 0] = -delta
    cash_position[:, 0] = -option_value[:, 0] + delta * S0
    portfolio_value[:, 0] = stock_position[:, 0] * S0 + cash_position[:, 0]

    for i in range(1, steps-1):

        S_current = paths[:, i] # Obecna cena indeksu
        t_to_maturity = T_years - i*dt # Czas do wygaśnięcia
        option_value[:, i] = option_price(S_current, K, t_to_maturity, r, sigma, option_type) # Aktualizacja wartości opcji
        cash_position[:, i] = cash_position[:, i - 1] * np.exp(r * dt) # Aktualizacja wartości gotówki (uwzględniamy oprocentowanie)
        portfolio_before = stock_position[:, i - 1] * S_current + cash_position[:, i] # Wartość portfela przed rebalansowaniem

        # Sprawdzenie, czy jest to punkt rehedgingu
        if i in hedge_indices:
            new_delta = option_delta(S_current, K, t_to_maturity, r, sigma, option_type)
            stock_position[:, i] = -new_delta
            cash_position[:, i] += (new_delta - (-stock_position[:, i - 1])) * S_current

        else:
            stock_position[:, i] = stock_position[:, i - 1]

        portfolio_after = stock_position[:, i] * S_current + cash_position[:, i] # Wartość portfela po rebalansowaniu
        portfolio_value[:, i] = portfolio_after

    # Moment wykonania opcji T = T_years
    S_T = paths[:,-1]
    option_value[:, -1] = option_price(S_T, K, 0, r, sigma, option_type)
    cash_position[:,-1] = cash_position[:, -2] * np.exp(r * dt)
    stock_position[:,-1] = stock_position[:,-2]
    portfolio_value[:, -1] = stock_position[:,-1] * S_T + cash_position[:,-1] + option_value[:,-1]

    portfolio_composition = {
        'stock': stock_position,
        'cash': cash_position,
        'option_value': option_value,
        'portfolio_value': portfolio_value
    }

    return portfolio_value[:,-1], portfolio_composition