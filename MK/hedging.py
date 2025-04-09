from math import gamma
import numpy as np
import pandas as pd
from .pricing import *
from scipy.stats import norm


def adjust_vanilla_delta(vanilla_delta, delta_margin):
    adjusted_delta = vanilla_delta.copy()
    mask_near_1 = np.abs(1 - vanilla_delta) < delta_margin
    adjusted_delta[mask_near_1] = 1
    mask_near_0 = np.abs(vanilla_delta) < delta_margin
    adjusted_delta[mask_near_0] = 0

    return adjusted_delta

def adjust_binary_delta(binary_delta, vanilla_delta, delta_margin):
    cutoff_binary_delta = np.where(
        (np.abs(vanilla_delta) > delta_margin) |
        (np.abs(1 - vanilla_delta) > delta_margin),
        binary_delta,
        0
    )

    return cutoff_binary_delta

def wallet_delta(paths, T_years, sigma, r, K, option_type='call', n_hedge=252):
    """
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
    portfolio_delta = np.zeros((N, steps))

    # Inicjalizacja portfela
    S0 = paths[:, 0]
    t_to_maturity = T_years
    option_value[:, 0] = option_price(S0, K, t_to_maturity, r, sigma, option_type)
    delta = option_delta(S0, K, t_to_maturity, r, sigma, option_type)

    # Początkowy skład portfela
    stock_position[:, 0] = delta
    cost = (np.abs(option_value[:, 0]) + np.abs(delta * S0)) * 0.004
    cash_position[:, 0] = (option_value[:, 0] - delta * S0) - cost
    portfolio_value[:, 0] = stock_position[:, 0] * S0 + cash_position[:, 0]
    portfolio_delta[:, 0] = -delta + stock_position[:, 0]

    for i in range(1, steps-1):

        S_current = paths[:, i] # Obecna cena indeksu
        t_to_maturity = T_years - i*dt # Czas do wygaśnięcia
        option_value[:, i] = option_price(S_current, K, t_to_maturity, r, sigma, option_type) # Aktualizacja wartości opcji
        cash_position[:, i] = cash_position[:, i - 1] * np.exp(r * dt) # Aktualizacja wartości gotówki (uwzględniamy oprocentowanie)
        new_delta = option_delta(S_current, K, t_to_maturity, r, sigma, option_type)

        # Sprawdzenie, czy jest to punkt rehedgingu
        if i in hedge_indices:
            stock_position[:, i] = new_delta
            cost = np.abs((stock_position[:, i] - (stock_position[:, i - 1])) * S_current) * 0.004
            cash_position[:, i] -= (stock_position[:, i] - (stock_position[:, i - 1])) * S_current + cost

        else:
            stock_position[:, i] = stock_position[:, i - 1]

        portfolio_after = stock_position[:, i] * S_current + cash_position[:, i] # Wartość portfela po rebalansowaniu
        portfolio_value[:, i] = portfolio_after
        portfolio_delta[:, i] = -new_delta + stock_position[:, i]

    # Moment wykonania opcji T = T_years
    S_T = paths[:,-1]
    option_value[:, -1] = option_price(S_T, K, 0, r, sigma, option_type)
    cash_position[:,-1] = cash_position[:, -2] * np.exp(r * dt)
    stock_position[:,-1] = stock_position[:,-2]
    portfolio_value[:, -1] = stock_position[:,-1] * S_T + cash_position[:,-1] - option_value[:,-1]
    portfolio_delta[:, -1] = 0

    portfolio_composition = {
        'stock': stock_position,
        'cash': cash_position,
        'option_value': option_value,
        'portfolio_value': portfolio_value
    }

    return portfolio_value[:,-1], portfolio_composition


def wallet_delta_2(paths, T_years, sigma, r, K, option_type='call', n_hedge=252):
    """
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
    portfolio_delta = np.zeros((N, steps))

    # Inicjalizacja portfela
    S0 = paths[:, 0]
    t_to_maturity = T_years
    option_value[:, 0] = option_price(S0, K, t_to_maturity, r, sigma, option_type)
    delta = option_delta(S0, K, t_to_maturity, r, sigma, option_type)

    # Początkowy skład portfela
    stock_position[:, 0] = delta
    cost = (np.abs(option_value[:, 0]) + np.abs(delta * S0)) * 0.004
    cash_position[:, 0] = (option_value[:, 0] - delta * S0) - cost
    portfolio_value[:, 0] = stock_position[:, 0] * S0 + cash_position[:, 0]
    portfolio_delta[:, 0] = -delta + stock_position[:, 0]

    margins = {'delta_margin': 0.01}


    for i in range(1, steps-1):

        S_current = paths[:, i]
        t_to_maturity = T_years - i*dt
        option_value[:, i] = option_price(S_current, K, t_to_maturity, r, sigma, option_type)
        cash_position[:, i] = cash_position[:, i - 1] * np.exp(r * dt)
        new_delta = option_delta(S_current, K, t_to_maturity, r, sigma, option_type)

        fixed_delta = adjust_vanilla_delta(new_delta, margins['delta_margin'])  #

        if i in hedge_indices:
            stock_position[:, i] = fixed_delta
            cost = np.abs((stock_position[:, i] - (stock_position[:, i - 1])) * S_current) * 0.004
            cash_position[:, i] -= (stock_position[:, i] - (stock_position[:, i - 1])) * S_current + cost

        else:
            stock_position[:, i] = stock_position[:, i - 1]

        portfolio_after = stock_position[:, i] * S_current + cash_position[:, i] # Wartość portfela po rebalansowaniu
        portfolio_value[:, i] = portfolio_after
        portfolio_delta[:, i] = -fixed_delta + stock_position[:, i]

    # Moment wykonania opcji T = T_years
    S_T = paths[:,-1]
    option_value[:, -1] = option_price(S_T, K, 0, r, sigma, option_type)
    cash_position[:,-1] = cash_position[:, -2] * np.exp(r * dt)
    stock_position[:,-1] = stock_position[:,-2]
    portfolio_value[:, -1] = stock_position[:,-1] * S_T + cash_position[:,-1] - option_value[:,-1]
    portfolio_delta[:, -1] = 0

    portfolio_composition = {
        'stock': stock_position,
        'cash': cash_position,
        'option_value': option_value,
        'portfolio_value': portfolio_value,
        'portfolio_delta': portfolio_delta
    }

    return portfolio_value[:, -1], portfolio_composition




def wallet_delta_gamma(paths, T_years, sigma, r, K, K_binary, option_type='call',
                       n_hedge=252, binary_option_type='binary_call'):

    N, steps = paths.shape
    dt = T_years / steps

    if n_hedge > steps:
        raise ValueError('Paths should be longer')

    hedge_frequency = steps / n_hedge
    hedge_frequency = max(1, int(hedge_frequency))
    hedge_indices = np.arange(0, steps, hedge_frequency)

    stock_position = np.zeros((N, steps))
    binary_position = np.zeros((N, steps))
    cash_position = np.zeros((N, steps))
    option_value = np.zeros((N, steps))
    binary_value = np.zeros((N, steps))
    portfolio_value = np.zeros((N, steps))
    portfolio_delta = np.zeros((N, steps))
    portfolio_gamma = np.zeros((N, steps))

    S0 = paths[:, 0]
    t_to_maturity = T_years

    # Wartość i greki oryginalnej opcji (którą hedgujemy)
    option_value[:, 0] = option_price(S0, K, t_to_maturity, r, sigma, option_type)
    vanilla_delta = option_delta(S0, K, t_to_maturity, r, sigma, option_type)
    vanilla_gamma = option_gamma(S0, K, t_to_maturity, r, sigma, option_type)

    # Wartość i greki opcji binarnej
    binary_value[:, 0] = binary_option_price(S0, K_binary, t_to_maturity, r, sigma, binary_option_type)
    binary_delta =  option_delta(S0, K_binary, t_to_maturity, r, sigma, binary_option_type)
    binary_gamma =  option_gamma(S0, K_binary, t_to_maturity, r, sigma, binary_option_type)

    # Inicjalizacja pozycji
    binary_position[:, 0] = vanilla_gamma / binary_gamma
    stock_position[:, 0] = vanilla_delta - binary_position[:, 0] * binary_delta

    cost = (np.abs(binary_position[:, 0] * binary_value[:, 0]) + np.abs(stock_position[:, 0] * S0)) * 0.004

    cash_position[:, 0] = (option_value[:, 0] - binary_position[:, 0] * binary_value[:, 0] - stock_position[:, 0] * S0) - cost

    # Wartość całego portfela
    portfolio_value[:, 0] = (cash_position[:, 0] +
                             stock_position[:, 0] * S0 +
                             binary_position[:, 0] * binary_value[:, 0] -
                             option_value[:, 0])

    # Delta i gamma całego portfela w momencie t=0
    portfolio_delta[:, 0] = -vanilla_delta + stock_position[:, 0] + binary_position[:, 0] * binary_delta
    portfolio_gamma[:, 0] = -vanilla_gamma + binary_position[:, 0] * binary_gamma

    margins = {'delta_margin': 0.001, 'binary_gamma_margin': 1e-3}
    for i in range(1, steps-1):
        S_current = paths[:, i]
        t_to_maturity = T_years - i * dt
        option_value[:, i] = option_price(S_current, K, t_to_maturity, r, sigma, option_type)
        binary_value[:, i] = binary_option_price(S_current, K_binary, t_to_maturity, r, sigma, binary_option_type)
        cash_position[:, i] = cash_position[:, i - 1] * np.exp(r * dt)

        vanilla_delta = option_delta(S_current, K, t_to_maturity, r, sigma, option_type)
        vanilla_gamma = option_gamma(S_current, K, t_to_maturity, r, sigma, option_type)
        binary_delta = option_delta(S_current, K_binary, t_to_maturity, r, sigma, binary_option_type)
        binary_gamma = option_gamma(S_current, K_binary, t_to_maturity, r, sigma, binary_option_type)

        # Obcinanie przypadków ITM i OTM
        fixed_vanilla_delta = adjust_vanilla_delta(vanilla_delta, margins['delta_margin']) #
        fixed_binary_delta = adjust_binary_delta(binary_delta, fixed_vanilla_delta, margins['delta_margin']) #
        do_calculations = (fixed_vanilla_delta != 0) | (fixed_vanilla_delta != 1)

        if i in hedge_indices:
            # Obsługa zer
            binary_gamma_safe = np.where(do_calculations,
                                   np.where(np.abs(binary_gamma) > margins['binary_gamma_margin'], binary_gamma, np.sign(binary_gamma) * margins['binary_gamma_margin']),
                                         1) # 1 jest arbitralne

            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratio = np.log(np.abs(vanilla_gamma)) - np.log(np.abs(binary_gamma_safe))
                gamma_ratio = np.sign(vanilla_gamma * binary_gamma_safe) * np.exp(log_ratio)


            new_delta_2 = gamma_ratio
            new_delta_1 = fixed_vanilla_delta - new_delta_2 * fixed_binary_delta

            stock_position[:, i] = new_delta_1
            binary_position[:, i] = new_delta_2

            cost = (np.abs((stock_position[:, i] - stock_position[:, i - 1]) * S_current) +
                np.abs(binary_position[:, i] - binary_position[:, i - 1]) * binary_value[:, i]) * 0.004

            cash_delta = ((stock_position[:, i] - stock_position[:, i - 1]) * S_current +
                          (binary_position[:, i] - binary_position[:, i - 1]) * binary_value[:, i])

            cash_position[:, i] -= cash_delta + cost

        else:
            stock_position[:, i] = stock_position[:, i - 1]
            binary_position[:, i] = binary_position[:, i - 1]

        portfolio_delta[:, i] = -fixed_vanilla_delta + stock_position[:, i] + binary_position[:, i] * fixed_binary_delta
        portfolio_gamma[:, i] = -vanilla_gamma + binary_position[:, i] * binary_gamma

        portfolio_value[:, i] = (cash_position[:, i] +
                                 stock_position[:, i] * S_current +
                                 binary_position[:, i] * binary_value[:, i] -
                                 option_value[:, i])

    # Moment wykonania opcji T = T_years
    S_T = paths[:, -1]
    stock_position[:, -1] = stock_position[:, -2]
    binary_position[:, -1] = binary_position[:, -2]
    cash_position[:, -1] = cash_position[:, -2] * np.exp(r * dt)
    option_value[:, -1] = option_price(S_T, K, 0, r, sigma, option_type)
    binary_value[:, -1] = binary_option_price(S_T, K_binary, 0, r, sigma, binary_option_type)
    portfolio_value[:, -1] = (cash_position[:, -1] + stock_position[:, -1] * S_T +
                              binary_position[:, -1] * binary_value[:, -1] -
                              option_value[:, -1])

    portfolio_delta[:, -1] = 0
    portfolio_gamma[:, -1] = 0

    # Wynik na koniec okresu
    final_pnl = portfolio_value[:, -1]

    portfolio_composition = {
        'stock': stock_position,
        'binary': binary_position,
        'cash': cash_position,
        'option_value': option_value,
        'portfolio_value': portfolio_value,
        'delta': portfolio_delta,
        'gamma': portfolio_gamma
    }
    return final_pnl, portfolio_composition


def wallet_delta_gamma_2(paths, T_years, sigma, r, K, K_binaries, option_type='call',
                       n_hedge=252):
    N, steps = paths.shape
    dt = T_years / steps

    if n_hedge > steps:
        raise ValueError('Paths should be longer')

    hedge_frequency = steps / n_hedge
    hedge_frequency = max(1, int(hedge_frequency))
    hedge_indices = np.arange(0, steps, hedge_frequency)

    stock_position = np.zeros((N, steps))
    binary_call_position = np.zeros((N, steps))
    binary_put_position = np.zeros((N, steps))
    cash_position = np.zeros((N, steps))
    option_value = np.zeros((N, steps))
    binary_call_value = np.zeros((N, steps))
    binary_put_value = np.zeros((N, steps))
    portfolio_value = np.zeros((N, steps))
    portfolio_delta = np.zeros((N, steps))
    portfolio_gamma = np.zeros((N, steps))

    S0 = paths[:, 0]
    t_to_maturity = T_years

    # Wartość i greki oryginalnej opcji (którą hedgujemy)
    option_value[:, 0] = option_price(S0, K, t_to_maturity, r, sigma, option_type)
    vanilla_delta = option_delta(S0, K, t_to_maturity, r, sigma, option_type)
    vanilla_gamma = option_gamma(S0, K, t_to_maturity, r, sigma, option_type)

    # Wartość i greki opcji binarnych
    # Call
    binary_call_value[:, 0] = binary_option_price(S0, K_binaries[0], t_to_maturity, r, sigma, 'binary_call')
    binary_call_delta = option_delta(S0, K_binaries[0], t_to_maturity, r, sigma, 'binary_call')
    binary_call_gamma = option_gamma(S0, K_binaries[0], t_to_maturity, r, sigma, 'binary_call')

    # Put
    binary_put_value[:, 0] = binary_option_price(S0, K_binaries[1], t_to_maturity, r, sigma, 'binary_put')
    binary_put_delta = option_delta(S0, K_binaries[0], t_to_maturity, r, sigma, 'binary_put')
    binary_put_gamma = option_gamma(S0, K_binaries[0], t_to_maturity, r, sigma, 'binary_put')

    # Inicjalizacja pozycji
    binary_call_position[:, 0] = (vanilla_gamma + K/S0) / binary_call_gamma
    binary_put_position[:, 0] = (K / S0) * binary_call_position[:, 0]
    stock_position[:, 0] = vanilla_delta - (vanilla_gamma / (binary_call_gamma - K/S0)) * (binary_call_delta + (K/S0) * binary_put_delta)

    cost = (np.abs(binary_call_position[:, 0] * binary_call_value[:, 0])
            + np.abs(stock_position[:, 0] * S0) + np.abs(binary_put_position[:, 0] * binary_put_value[:, 0])) * 0.004

    cash_position[:, 0] = (option_value[:, 0] -
                           binary_call_position[:, 0] * binary_call_value[:, 0] -
                           stock_position[:, 0] * S0 - binary_put_position[:, 0] * binary_put_value[:, 0]) - cost

    # Wartość całego portfela
    portfolio_value[:, 0] = (cash_position[:, 0] +
                             stock_position[:, 0] * S0 +
                             binary_call_position[:, 0] * binary_call_value[:, 0] +
                             binary_put_position[:, 0] * binary_put_value[:, 0] -
                             option_value[:, 0])

    # Delta i gamma całego portfela w momencie t=0
    portfolio_delta[:, 0] = (-vanilla_delta + stock_position[:, 0] +
                             binary_call_position[:, 0] * binary_call_delta +
                             binary_put_position[:, 0] * binary_put_delta)

    portfolio_gamma[:, 0] = (-vanilla_gamma + binary_call_position[:, 0] * binary_call_gamma
                             + binary_put_position[:, 0] * binary_put_gamma)

    margins = {'delta_margin': 0.001, 'binary_gamma_margin': 1e-3}
    for i in range(1, steps - 1):
        S_current = paths[:, i]
        t_to_maturity = T_years - i * dt
        option_value[:, i] = option_price(S_current, K, t_to_maturity, r, sigma, option_type)
        binary_call_value[:, i] = binary_option_price(S0, K_binaries[0], t_to_maturity, r, sigma, 'binary_call')
        binary_put_value[:, i] = binary_option_price(S0, K_binaries[1], t_to_maturity, r, sigma, 'binary_put')
        cash_position[:, i] = cash_position[:, i - 1] * np.exp(r * dt)

        vanilla_delta = option_delta(S_current, K, t_to_maturity, r, sigma, option_type)
        vanilla_gamma = option_gamma(S_current, K, t_to_maturity, r, sigma, option_type)
        binary_call_delta = option_delta(S_current, K_binaries[0], t_to_maturity, r, sigma, 'binary_call')
        binary_call_gamma = option_gamma(S_current, K_binaries[0], t_to_maturity, r, sigma, 'binary_call')
        binary_put_delta = option_delta(S_current, K_binaries[1], t_to_maturity, r, sigma, 'binary_put')
        binary_put_gamma = option_gamma(S_current, K_binaries[1], t_to_maturity, r, sigma, 'binary_put')

        # Obcinanie przypadków ITM i OTM
        fixed_vanilla_delta = adjust_vanilla_delta(vanilla_delta, margins['delta_margin'])  #
        fixed_binary_call_delta = adjust_binary_delta(binary_call_delta, fixed_vanilla_delta, margins['delta_margin'])  #
        fixed_binary_put_delta = adjust_binary_delta(binary_put_delta, fixed_vanilla_delta, margins['delta_margin'])  #
        do_calculations = (fixed_vanilla_delta != 0) | (fixed_vanilla_delta != 1)

        if i in hedge_indices:
            # Obsługa zer
            '''binary_gamma_safe = np.where(do_calculations,
                                         np.where(np.abs(binary_gamma) > margins['binary_gamma_margin'], binary_gamma,
                                                  np.sign(binary_gamma) * margins['binary_gamma_margin']),
                                         1)  # 1 jest arbitralne

            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratio = np.log(np.abs(vanilla_gamma)) - np.log(np.abs(binary_gamma_safe))
                gamma_ratio = np.sign(vanilla_gamma * binary_gamma_safe) * np.exp(log_ratio)'''


            binary_call_position[:, i] = (vanilla_gamma + K / S_current) / binary_call_gamma
            binary_put_position[:, i] = (K / S_current) * binary_call_position[:, i]
            stock_position[:, i] = (fixed_vanilla_delta - (vanilla_gamma / (binary_call_gamma - K / S_current)) *
                                    (fixed_binary_call_delta + (K / S_current) * fixed_binary_put_delta))

            cost = (np.abs(binary_call_position[:, i] * binary_call_value[:, i])
                    + np.abs(stock_position[:, i] * S_current) + np.abs(
                        binary_put_position[:, i] * binary_put_value[:, i])) * 0.004

            cash_delta = ((stock_position[:, i] - stock_position[:, i - 1]) * S_current +
                          (binary_call_position[:, i] - binary_call_position[:, i - 1]) * binary_call_value[:, i] +
                          (binary_put_position[:, i] - binary_put_position[:, i - 1]) * binary_put_value[:, i])

            cash_position[:, i] -= cash_delta + cost

        else:
            stock_position[:, i] = stock_position[:, i - 1]
            binary_call_position[:, i] = binary_call_position[:, i - 1]
            binary_put_position[:, i] = binary_put_position[:, i - 1]


        portfolio_value[:, i] = (cash_position[:, i] +
                                 stock_position[:, i] * S_current +
                                 binary_call_position[:, i] * binary_call_value[:, i] +
                                 binary_put_position[:, i] * binary_put_value[:, i] -
                                 option_value[:, i])

        # Delta i gamma całego portfela w momencie t=0
        portfolio_delta[:, i] = (-vanilla_delta + stock_position[:, i] +
                                 binary_call_position[:, i] * fixed_binary_call_delta +
                                 binary_put_position[:, i] * fixed_binary_put_delta)

        portfolio_gamma[:, i] = (-vanilla_gamma + binary_call_position[:, i] * binary_call_gamma
                                 + binary_put_position[:, i] * binary_put_gamma)

    # Moment wykonania opcji T = T_years
    S_T = paths[:, -1]
    stock_position[:, -1] = stock_position[:, -2]
    binary_call_position[:, -1] = binary_call_position[:, -2]
    binary_put_position[:, -1] = binary_put_position[:, -2]
    cash_position[:, -1] = cash_position[:, -2] * np.exp(r * dt)
    option_value[:, -1] = option_price(S_T, K, 0, r, sigma, option_type)
    binary_call_value[:, -1] = binary_option_price(S_T, K_binaries[0], 0, r, sigma, 'binary_call')
    binary_put_value[:, -1] = binary_option_price(S_T, K_binaries[1], 0, r, sigma, 'binary_put')
    portfolio_value[:, -1] = (cash_position[:, -1] +
                                 stock_position[:, -1] * S_T +
                                 binary_call_position[:, -1] * binary_call_value[:, -1] +
                                 binary_put_position[:, -1] * binary_put_value[:, -1] -
                                 option_value[:, -1])

    portfolio_delta[:, -1] = 0
    portfolio_gamma[:, -1] = 0

    # Wynik na koniec okresu
    final_pnl = portfolio_value[:, -1]

    portfolio_composition = {
        'stock': stock_position,
        'binary_call': binary_call_position,
        'binary_put': binary_put_position,
        'cash': cash_position,
        'option_value': option_value,
        'portfolio_value': portfolio_value,
        'delta': portfolio_delta,
        'gamma': portfolio_gamma
    }

    return final_pnl, portfolio_composition
