import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def generate_gbm_paths(N, T, s0, mean, volatility, h=1, plot=False):
    """
    Generuje ścieżki ceny przy użyciu geometrycznego ruchu Browna.

    Parameters:
    -----------
    N : int
        Liczba ścieżek.
    T : int
        Horyzont czasowy (liczba lat).
    s0 : float
        Początkowa cena aktywa.
    mean : float
        Średnia logarytmicznych zwrotów rocznych.
    volatility : float
        Odchylenie standardowe logarytmicznych zwrotów rocznych.
    h : float, optional
        Krok czasowy w dniach. Domyślnie 1 dzień.
    plot : bool, optional
        Czy wykreślić ścieżki. Domyślnie False.

    Returns:
    --------
    paths : ndarray
        Macierz ścieżek cenowych (N x (T / dt)).
    time : ndarray
        Macierz punktów czasowych.
    """
    dt = h / 252  # Konwersja kroku dziennego na część roku (252 dni handlowych)
    n = int(T / dt)

    # Przeskalowane parametry dla danego kroku czasowego
    mu_dt = mean * dt
    sigma_dt = volatility * np.sqrt(dt)

    # Generowanie przyrostów logarytmicznych dla GBM
    increments = np.exp((mu_dt - sigma_dt ** 2 / 2) +
                        sigma_dt * np.random.normal(0, 1, size=(N, n-1)))

    # Dodanie punktu początkowego i obliczenie ścieżek cenowych
    paths = np.hstack([np.ones((N, 1)), increments])
    paths = s0 * paths.cumprod(axis=1)

    time = np.linspace(0, T, n)
    matrixtime = np.full(shape=(N, n), fill_value=time)

    if plot is True:
        plt.figure(figsize=(12, 8))
        plt.plot(matrixtime.T, paths.T, alpha=0.3)
        plt.title(f'Symulowane ścieżki ({N}) - Geometryczny ruch Browna')
        plt.xlabel(f'Czas (lata)')
        plt.ylabel('Wartość aktywa')

    return paths, matrixtime


def bootstrap_paths(N, T, s0, historical_data, plot=False):
    """
    Generuje ścieżki cenowe metodą bootstrapu z historycznych zwrotów.

    Parameters:
    -----------
    N : int
        Liczba ścieżek.
    T : int
        Horyzont czasowy (liczba dni).
    s0 : float
        Początkowa cena aktywa.
    historical_data : DataFrame
        Dane historyczne z cenami.
    plot : bool, optional
        Czy wykreślić ścieżki. Domyślnie False.

    Returns:
    --------
    paths : ndarray
        Macierz ścieżek cenowych.
    time : ndarray
        Macierz punktów czasowych.
    """
    T = 252*T
    sample = np.log(historical_data.Close / historical_data.Close.shift(1)).dropna()  # log returns
    rng = np.random.default_rng()  # Inicjalizacja generatora
    rates = np.exp(rng.choice(sample, size=(N, T), replace=True, shuffle=False))

    # Kumulowanie zwrotów
    paths = s0 * np.cumprod(rates, axis=1)
    paths = np.hstack([np.ones((N, 1)) * s0, paths])  # Dodaj początkową cenę

    time = np.arange(0, T + 1)
    matrixtime = np.full(shape=(N, T + 1), fill_value=time)

    if plot is True:
        plt.figure(figsize=(12, 8))
        plt.plot(matrixtime.T, paths.T, alpha=0.3)
        plt.title(f'Symulowane ścieżki ({N}) - Bootstrap')
        plt.xlabel('Czas (dni)')
        plt.ylabel('Wartość WIG20')
        plt.show()

    return paths, matrixtime


def generate_correlated_gbm_paths(N, T, s0_vec, mean_vec, vol_vec, corr_matrix, h=1):
    """
    Generuje ścieżki cen dla dwóch lub więcej aktywów przy użyciu skorelowanego geometrycznego ruchu Browna.

    Parameters:
    -----------
    N : int
        Liczba ścieżek.
    T : int
        Horyzont czasowy (liczba dni).
    s0_vec : list or ndarray
        Wektor początkowych cen aktywów.
    mean_vec : list or ndarray
        Wektor średnich logarytmicznych zwrotów dziennych.
    vol_vec : list or ndarray
        Wektor zmienności (odchylenie standardowe logarytmicznych zwrotów dziennych).
    corr_matrix : ndarray
        Macierz korelacji między aktywami.
    h : float, optional
        Krok czasowy. Domyślnie 1 dzień.

    Returns:
    --------
    paths : dict
        Słownik zawierający macierze ścieżek cenowych dla każdego aktywa.
    time : ndarray
        Macierz punktów czasowych.
    """
    dt = h
    n = int(T / h)
    num_assets = len(s0_vec)

    # Przekształcenie wektorów na tablice numpy
    s0_vec = np.array(s0_vec)
    mean_vec = np.array(mean_vec)
    vol_vec = np.array(vol_vec)

    # Parametry dla procesu GBM
    mu = mean_vec * h
    sigma = vol_vec * np.sqrt(h)

    # Dekompozycja Choleskiego macierzy korelacji
    L = np.linalg.cholesky(corr_matrix)

    # Inicjalizacja ścieżek
    paths = {}
    for i in range(num_assets):
        paths[f'asset_{i}'] = np.zeros((N, n + 1))
        paths[f'asset_{i}'][:, 0] = s0_vec[i]

    # Generowanie ścieżek
    for t in range(1, n + 1):
        # Generowanie nieskorelowanych losowych liczb
        Z = np.random.normal(0, 1, size=(N, num_assets))

        # Korelacja liczb losowych
        correlated_Z = Z @ L.T

        # Aktualizacja cen dla każdego aktywa
        for i in range(num_assets):
            paths[f'asset_{i}'][:, t] = paths[f'asset_{i}'][:, t - 1] * np.exp(
                (mu[i] - 0.5 * sigma[i] ** 2) * dt + sigma[i] * correlated_Z[:, i]
            )

    # Przygotowanie macierzy czasu
    time = np.linspace(0, T, n + 1)



    return paths, time

