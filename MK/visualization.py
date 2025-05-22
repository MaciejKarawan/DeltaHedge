import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib.dates import DateFormatter, MonthLocator
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.ticker import FuncFormatter
from scipy import stats as scipy_stats
from .hedging import wallet_delta, wallet_delta_2, wallet_delta_gamma, wallet_delta_gamma_2, wallet_delta_gamma_3
from .simulation import generate_gbm_paths



def setup_plotting_style():
    """Konfiguruje styl wykresów dla całego projektu."""
    plt.rc("figure",
           autolayout=True,
           figsize=(11.69, 8.27),
           titlesize=10,
           titleweight='bold')
    plt.rc("axes",
           labelweight="bold",
           labelsize="large",
           titleweight="bold",
           titlesize=16,
           titlepad=10,
           titlelocation='center')

    sns.set_theme(style='whitegrid', context='paper', palette='dark')


def plot_price_history(data, title=None, highlight_dates=None, save=False):
    """
    Wykreśla historyczne ceny z opcjonalnym zaznaczeniem ważnych dat.

    Parameters:
    -----------
    data : DataFrame
        Dane cenowe z kolumną 'Date' i 'Close'.
    title : str, optional
        Tytuł wykresu.
    highlight_dates : dict, optional
        Słownik z datami do zaznaczenia i ich opisami.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Konwersja dat
    data_dt = pd.to_datetime(data.Date)

    # Wykres cen
    ax.plot(data_dt, data.Close, label="WIG20")

    # Formatowanie osi X
    #ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    #ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    # Zaznaczenie ważnych dat
    if highlight_dates:
        for date_str, desc in highlight_dates.items():
            date = pd.to_datetime(date_str)
            ax.axvline(date, color='red', linestyle='dotted', linewidth=1.5)
            ax.text(date + pd.Timedelta(days=10), data.Close.min(), desc,
                    color="red", rotation=90, verticalalignment="bottom", fontsize=10)

    # Formatowanie
    plt.xticks(rotation=45)
    #ax.set_xlabel("Data")
    ax.set_ylabel("WIG20")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"WIG20 {data.Date.iloc[0].strftime('%Y-%m-%d')} do {data.Date.iloc[-1].strftime('%Y-%m-%d')}")

    #plt.legend()
    plt.tight_layout()

    if save: plt.savefig('historic_wig20.png', format='png', bbox_inches='tight', dpi=300)

    plt.show()


def compare_returns_distribution(returns, returns2, save=False):
    """
    Porównuje rozkłady zwrotów z rozkładem normalnym.

    Parameters:
    -----------
    returns : Series
        Zwroty 1.
    log_returns : Series
        Zwroty 2.
    """
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))
    fig.suptitle('Porównanie zwrotów dziennych \n '
                 'dla okresu 01.01.2022 - 31.12.2022')

    # Scatterploty
    sns.scatterplot(x=range(len(returns)), y=returns, ax=ax[0, 0], color='red', label="Zwroty rzeczywiste")
    sns.scatterplot(x=range(len(returns2)), y=returns2, ax=ax[1, 0], color='blue', label="Zwroty symulowane")

    # Porównanie z rozkładem normalnym
    x = np.linspace(-4, 4, 100)
    ax[0, 1].plot(x, norm.pdf(x, 0, 1), 'r', label="N(0,1) PDF")
    ax[1, 1].plot(x, norm.pdf(x, 0, 1), 'r', label="N(0,1) PDF")

    # Znormalizowane histogramy
    sns.histplot((returns - returns.mean()) / returns.std(), kde=True, color='red', ax=ax[0, 1], label="Zwroty rzeczywiste")
    sns.histplot((returns2 - returns2.mean()) / returns2.std(), kde=True, color='blue', ax=ax[1, 1],
                 label="Zwroty symulowane")

    # Tytuły i legendy
    ax[0, 0].set_title("Zwroty rzeczywiste")
    ax[1, 0].set_title("Zwroty symulowane za pomocą metody Bootstrap")
    ax[0, 1].set_title("Histogram zwrotów vs N(0,1)")
    ax[1, 1].set_title("Histogram zwrotów vs N(0,1)")

    for i in range(2):
        for j in range(2):
            ax[i, j].legend()
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')

    ax[0, 1].set_ylim(0, 50)
    ax[1, 1].set_ylim(0, 50)
    ax[0, 1].set_xlim(-6, 4.5)
    ax[1, 1].set_xlim(-6, 4.5)

    ax[0, 0].set_ylim(-0.15, 0.1)
    ax[1, 0].set_ylim(-0.15, 0.1)

    plt.tight_layout()
    if save: plt.savefig('return_compare_bootstrap.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_historical_volatility(data, n_years_list, window_size=252, highlight_2022=True, plot=True, save=False):
    """
    Wykreśla zmienność historyczną dla różnych okresów czasu.

    Parameters:
    -----------
    data : DataFrame
        Dane cenowe z kolumnami 'Date' i 'Close'.
    n_years_list : list
        Lista liczb lat branych pod uwagę do obliczenia zmienności.
    window_size : int, optional
        Liczba dni handlowych w roku (domyślnie 252).
    highlight_2022 : bool, optional
        Czy dodać poziomą linię oznaczającą zmienność w 2022 roku.
    """
    # Konwersja dat i sortowanie danych od najstarszych do najnowszych
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Obliczanie zwrotów logarytmicznych
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()

    # Przygotowanie wykresu
    if plot is True: plt.figure(figsize=(8, 5))

    for n_years in n_years_list:
        # Obliczanie rozmiaru okna w dniach handlowych
        lookback_period = int(n_years * window_size)

        if lookback_period >= len(data):
            print(f"Ostrzeżenie: Okres {n_years} lat przekracza dostępne dane. Pomijam.")
            continue

        # Obliczanie rocznej zmienności za pomocą rolling window
        # Mnożymy przez sqrt(window_size) aby uzyskać roczną zmienność
        data[f'volatility_{n_years}y'] = data['log_return'].rolling(
            window=lookback_period
        ).std() * np.sqrt(window_size) * 100  # Wynik w procentach

        if plot is True:
            # Wykreślanie zmienności
            plt.plot(
                data['Date'],
                data[f'volatility_{n_years}y'],
                label=f'{n_years} {"rok" if n_years == 1 else "lata" if 1 < n_years < 5 else "lat"}'
        )

    # Dodanie poziomej linii dla zmienności w 2022 roku
    if highlight_2022 and plot is True:
        # Filtrowanie danych tylko dla 2022 roku
        data_2022 = data[(data['Date'] >= '2022-01-01') & (data['Date'] <= '2022-12-31')]

        if not data_2022.empty:
            # Obliczanie rocznej zmienności dla całego roku 2022
            volatility_2022 = data_2022['log_return'].std() * np.sqrt(window_size) * 100

            # Określenie zakresu dat dla linii poziomej (tylko 2022 rok)
            start_date = pd.Timestamp('2022-01-01')
            end_date = pd.Timestamp('2022-12-31')

            # Dodanie linii poziomej na wykresie
            plt.hlines(
                y=volatility_2022,
                xmin=start_date,
                xmax=end_date,
                colors='red',
                linestyles='dashed',
                linewidth=2,
                label=f'Zmienność w 2022: {volatility_2022:.2f}%'
            )

            # Dodanie adnotacji
            plt.annotate(
                f'{volatility_2022:.2f}%',
                xy=(end_date, volatility_2022),
                xytext=(10, 0),
                textcoords='offset points',
                color='red',
                fontweight='bold'
            )

            print(f"Zmienność w roku 2022: {volatility_2022:.2f}%")
        else:
            print("Brak danych dla roku 2022")

    if plot is True:
        plt.title(f'Wykres kroczącego odchylenia standardowego rocznego dla WIG20 dla różnych okresów')
        #plt.xlabel('Data')
        plt.ylabel('Historyczna zmienność roczna (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Okres obliczania:')

        # Formatowanie osi X
        plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save: plt.savefig('historic_wig20_volatility_2014.png', format='png', bbox_inches='tight', dpi=300)


        plt.show()

    return data  # Zwracamy dane z obliczonymi zmiennościami


def plot_historical_mean_returns(data, n_years_list, window_size=252, highlight_2022=True, plot=True, save=False):
    """
    Wykreśla średnie zwroty dla różnych okresów czasu.

    Parameters:
    -----------
    data : DataFrame
        Dane cenowe z kolumnami 'Date' i 'Close'.
    n_years_list : list
        Lista liczb lat branych pod uwagę do obliczenia średnich zwrotów.
    window_size : int, optional
        Liczba dni handlowych w roku (domyślnie 252).
    highlight_2022 : bool, optional
        Czy dodać poziomą linię oznaczającą średni zwrot w 2022 roku.
    """
    # Konwersja dat i sortowanie danych od najstarszych do najnowszych
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Obliczanie zwrotów logarytmicznych
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()

    # Przygotowanie wykresu
    if plot is True: plt.figure(figsize=(8, 5))

    for n_years in n_years_list:
        # Obliczanie rozmiaru okna w dniach handlowych
        lookback_period = int(n_years * window_size)

        if lookback_period >= len(data):
            print(f"Ostrzeżenie: Okres {n_years} lat przekracza dostępne dane. Pomijam.")
            continue

        # Obliczanie średnich rocznych zwrotów za pomocą rolling window
        # Mnożymy przez window_size aby uzyskać roczny zwrot
        data[f'mean_return_{n_years}y'] = data['log_return'].rolling(
            window=lookback_period
        ).mean() * window_size * 100  # Wynik w procentach (annualizowany)

        if plot is True:
            # Wykreślanie średnich zwrotów
            plt.plot(
                data['Date'],
                data[f'mean_return_{n_years}y'],
                label=f'{n_years} {"rok" if n_years == 1 else "lata" if 1 < n_years < 5 else "lat"}',
                linewidth=0.7
            )


    if highlight_2022 and plot is True:
        # Filtrowanie danych tylko dla 2022 roku
        data_2022 = data[(data['Date'] >= '2022-01-01') & (data['Date'] <= '2022-12-31')]

        if not data_2022.empty:
            # Obliczanie średniego rocznego zwrotu dla całego roku 2022
            mean_return_2022 = data_2022['log_return'].mean() * window_size * 100

            # Określenie zakresu dat dla linii poziomej (tylko 2022 rok)
            start_date = pd.Timestamp('2022-01-01')
            end_date = pd.Timestamp('2022-12-31')

            # Dodanie linii poziomej na wykresie
            plt.hlines(
                y=mean_return_2022,
                xmin=start_date,
                xmax=end_date,
                colors='red',
                linestyles='dashed',
                linewidth=2,
                label=f'Średni zwrot w 2022: {mean_return_2022:.2f}%'
            )

            # Dodanie adnotacji
            plt.annotate(
                f'{mean_return_2022:.2f}%',
                xy=(end_date, mean_return_2022),
                xytext=(10, 0),
                textcoords='offset points',
                color='red',
                fontweight='bold'
            )

            print(f"Średni zwrot w roku 2022: {mean_return_2022:.2f}%")
        else:
            print("Brak danych dla roku 2022")

    if plot is True:
        plt.title(f'Wykres kroczącej średniej dla rocznego zwrotu WIG20 dla różnych okresów')
        #plt.xlabel('Data')
        plt.ylabel('Średni roczny zwrot (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Okres obliczania:')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Linia na poziomie 0%

        # Formatowanie osi X
        plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save: plt.savefig('historic_wig20_mean.png', format='png', bbox_inches='tight', dpi=300)

        plt.show()

    return data  # Zwracamy dane z obliczonymi średnimi zwrotami


def plot_quantiles_vs_historical(paths_list, historical_data, labels, colors, plot_together=True,
                                 title="Kwantyle symulowanych ścieżek vs rzeczywista trajektoria", save=True):
    """
    Wykreśla kwantyle symulowanych ścieżek na tle historycznych danych.
    Pozwala na elastyczne wyświetlanie wielu metod symulacji razem lub osobno.

    Parameters:
    -----------
    paths_list : list of ndarray
        Lista macierzy symulowanych ścieżek (każda macierz reprezentuje inną metodę).
    historical_data : DataFrame
        Dane historyczne.
    labels : list of str
        Lista etykiet dla każdej metody.
    colors : list of str
        Lista kolorów dla każdej metody.
    plot_together : bool, optional
        Czy wykreślać wszystkie metody na jednym wykresie (True) czy osobno (False).
    title : str, optional
        Tytuł wykresu.
    """
    if not isinstance(paths_list, list):
        paths_list = [paths_list]
        labels = [labels]
        colors = [colors]

    if plot_together:
        # Wszystkie metody na jednym wykresie
        plt.figure(figsize=(8, 5))

        # Wykreślanie danych historycznych
        plt.plot(np.arange(0, len(historical_data)), historical_data.Close, label='Dane historyczne', color='red')

        # Wykreślanie każdej metody
        for paths, label, color in zip(paths_list, labels, colors):
            quantiles_paths = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
            time = np.arange(paths.shape[1])

            # Mediana
            plt.plot(time, quantiles_paths[2], label=f"Mediana {label}", color=color)

            # Przedziały kwantylowe
            plt.fill_between(time, quantiles_paths[1], quantiles_paths[3], color=color, alpha=0.2,
                             label=f"IQR (25%-75%) {label}")

            # Ekstremalne kwantyle
            plt.plot(time, quantiles_paths[0], color=color, ls=':', label=f'5% / 95% {label}')
            plt.plot(time, quantiles_paths[4], color=color, ls=':')

        plt.legend()
        plt.xlabel("Czas (dni)")
        plt.ylabel("WIG20")
        plt.title(title)
        plt.tight_layout()
        if save: plt.savefig('compare_gbm_bootstrap_historic_T2.png', format='png', bbox_inches='tight', dpi=300)
        plt.show()

    else:
        # Każda metoda na osobnym wykresie
        for i, (paths, label, color) in enumerate(zip(paths_list, labels, colors)):
            quantiles_paths = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
            time = np.arange(paths.shape[1])

            plt.figure(figsize=(8, 5))

            # Mediana
            plt.plot(time, quantiles_paths[2], label=f"Mediana {label}", color=color)

            # Dane historyczne
            plt.plot(np.arange(0, len(historical_data)), historical_data.Close, label='Dane historyczne', color='red')

            # Przedziały kwantylowe
            plt.fill_between(time, quantiles_paths[1], quantiles_paths[3], color=color, alpha=0.2,
                             label=f"IQR (25%-75%) {label}")

            # Ekstremalne kwantyle
            plt.plot(time, quantiles_paths[0], color=color, ls=':', label=f'5% {label}')
            plt.plot(time, quantiles_paths[4], color=color, ls=':', label=f'95% {label}')

            plt.legend()
            plt.xlabel("Czas (dni)")
            plt.ylabel("WIG20")
            plt.title(f"{title} - {label}")
            plt.tight_layout()
            if save: plt.savefig(f'quantile_{label}.png', format='png', bbox_inches='tight', dpi=300)
            plt.show()


def plot_option_payoff_histogram(option_type, strike, paths, last_price, r=0.05, T=1, bins=50, figsize=(7, 6), save=False):
    """
    Rysuje histogram wypłat (payoff) dla opcji europejskich na podstawie symulowanych ścieżek ceny aktywa.

    Parametry:
    -----------
    option_type : str
        Typ opcji - 'call' lub 'put'
    strike : float
        Cena wykonania opcji (strike price)
    paths : numpy.ndarray
        Macierz symulowanych ścieżek ceny aktywa o wymiarach (N, T+1),
        gdzie N to liczba ścieżek, T+1 to liczba momentów czasowych (włącznie z t=0).
        Ostatnia kolumna zawiera ceny aktywa w terminie wygaśnięcia opcji.
    r : float, optional
        Stopa wolna od ryzyka (w formacie dziesiętnym, np. 0.05 dla 5%), domyślnie 0.05
    T : float, optional
        Czas do wygaśnięcia opcji w latach, domyślnie 1
    bins : int, optional
        Liczba przedziałów w histogramie, domyślnie 50
    figsize : tuple, optional
        Rozmiar wykresu, domyślnie (12, 8)

    Zwraca:
    --------
    fig : matplotlib.figure.Figure
        Figura z wykresami
    """
    # Sprawdzenie poprawności parametrów
    if option_type.lower() not in ['call', 'put']:
        raise ValueError("Typ opcji musi być 'call' lub 'put'")

    # Pobierz ceny aktywa w terminie wygaśnięcia (ostatnia kolumna macierzy paths)
    final_prices = paths[:, -1]

    # Oblicz wypłaty opcji
    if option_type.lower() == 'call':
        payoffs = np.maximum(final_prices - strike, 0)
        option_name = f"Call_{strike}"
    else:  # put
        payoffs = np.maximum(strike - final_prices, 0)
        option_name = f"Put_{strike}"

    # Oblicz zdyskontowane wypłaty (wartość obecna wypłat)
    discounted_payoffs = payoffs * np.exp(-r * T)

    # Statystyki
    mean_payoff = np.mean(payoffs)
    mean_discounted = np.mean(discounted_payoffs)
    max_payoff = np.max(payoffs)
    positive_payoffs = np.sum(payoffs > 0) / len(payoffs) * 100  # procent opcji in-the-money

    # Przygotuj wykres
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Histogram cen końcowych aktywa
    sns.histplot(final_prices, bins=bins, kde=True, ax=axes[0])
    axes[0].axvline(strike, color='red', linestyle='--', label=f'Strike = {strike}')
    axes[0].set_title(f'Rozkład końcowych cen aktywa bazowego', fontsize=11)
    axes[0].set_xlabel('Cena aktywa', fontsize=7.5)
    axes[0].set_ylabel('Częstość', fontsize=7.5)
    axes[0].legend()

    # Histogram wypłat opcji
    sns.histplot(payoffs, bins=bins, kde=True, ax=axes[1])
    axes[1].axvline(mean_payoff, color='green', linestyle='--',
                    label=f'Średnia wypłata = {mean_payoff:.2f}')
    axes[1].set_title(f'Rozkład wypłat {option_name}, ostatnia cena {last_price} zł', fontsize=11)
    axes[1].set_xlabel('Payoff', fontsize=7.5)
    axes[1].set_ylabel('Częstość', fontsize=7.5)
    axes[1].legend()

    # Dodanie tekstowego opisu statystyk
    plt.figtext(0.02, 0.02, f"Statystyki:\n"
                            f"- Maksymalna wypłata: {max_payoff:.4f}\n"
                            f"- Opcje in-the-money: {positive_payoffs:.2f}%", fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Zrób miejsce na tekst

    if save: plt.savefig(f'payoff_{option_type}_{strike}.png', format='png', bbox_inches='tight', dpi=300)

    return fig


def plot_correlation_analysis(data, title=None, save=False):
    """
    Tworzy wizualizacje korelacji między zwrotami WIG20 i KGHM.

    Parameters:
    -----------
    data : DataFrame
        Dane z kolumnami 'WIG20_log_return' i 'KGHM_log_return'.
    title : str, optional
        Tytuł wykresu.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # 1. Scatter plot logarytmicznych zwrotów
    sns.scatterplot(x='WIG20_log_return', y='KGHM_log_return', data=data, ax=ax)

    ax.set_xlabel('WIG20 log return')
    ax.set_ylabel('KGHM log return')

    # Dodanie linii trendu
    x = data['WIG20_log_return']
    y = data['KGHM_log_return']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--")

    correlation = data[['WIG20_log_return', 'KGHM_log_return']].corr().iloc[0, 1]
    if title:
        fig.suptitle(f"{title}\nWspółczynnik korelacji: {correlation:.4f}")
    else:
        fig.suptitle(f"Analiza korelacji między WIG20 i KGHM\nWspółczynnik korelacji: {correlation:.4f}", size=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if save: plt.savefig('qq_kghm_wig20.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()

    return correlation


def plot_autocorrelation(data, lags=30):
    """
    Wykreśla autokorelację i wzajemną korelację zwrotów WIG20 i KGHM.

    Parameters:
    -----------
    data : DataFrame
        Dane z kolumnami 'WIG20_log_return' i 'KGHM_log_return'.
    lags : int, optional
        Liczba opóźnień do analizy. Domyślnie 30.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 10))

    # Autocorrelation WIG20
    pd.plotting.autocorrelation_plot(data['WIG20_log_return'], ax=axs[0])
    axs[0].set_title('Autokorelacja zwrotów WIG20')
    axs[0].set_xlim(0, lags)

    # Autocorrelation KGHM
    pd.plotting.autocorrelation_plot(data['KGHM_log_return'], ax=axs[1])
    axs[1].set_title('Autokorelacja zwrotów KGHM')
    axs[1].set_xlim(0, lags)

    plt.tight_layout()
    fig.suptitle('Analiza autokorelacji WIG20 i KGHM', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_2d_paths(paths_asset1, paths_asset2, time, N, sample_paths=5):
    """
    Wykreśla ścieżki dwóch aktywów na wykresie 2D.

    Parameters:
    -----------
    paths_asset1 : ndarray
        Macierz ścieżek dla pierwszego aktywa.
    paths_asset2 : ndarray
        Macierz ścieżek dla drugiego aktywa.
    time : ndarray
        Macierz punktów czasowych.
    N : int
        Liczba wszystkich ścieżek.
    sample_paths : int, optional
        Liczba próbek ścieżek do wyświetlenia. Domyślnie 5.
    """
    # Wybierz losowe indeksy ścieżek
    indices = np.random.choice(range(N), min(sample_paths, N), replace=False)

    # Przygotuj wykres
    plt.figure(figsize=(12, 10))

    # Wykres 2D ścieżek
    for idx in indices:
        plt.plot(paths_asset1[idx, :], paths_asset2[idx, :], alpha=0.7,
                 label=f'Ścieżka {idx + 1}')

    plt.scatter(paths_asset1[indices, 0], paths_asset2[indices, 0], color='green', s=50,
                label='Punkty początkowe')
    plt.scatter(paths_asset1[indices, -1], paths_asset2[indices, -1], color='red', s=50,
                label='Punkty końcowe')

    plt.title('Symulowane ścieżki dwóch aktywów')
    plt.xlabel('Aktywo 1')
    plt.ylabel('Aktywo 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Wykres 3D ścieżek (2 aktywa + czas)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for idx in indices:
        ax.plot(time, paths_asset1[idx, :], paths_asset2[idx, :], alpha=0.7,
                label=f'Ścieżka {idx + 1}')

    ax.scatter(np.zeros(len(indices)), paths_asset1[indices, 0], paths_asset2[indices, 0],
               color='green', s=50, label='Punkty początkowe')
    ax.scatter(np.ones(len(indices)) * time[-1], paths_asset1[indices, -1], paths_asset2[indices, -1],
               color='red', s=50, label='Punkty końcowe')

    ax.set_title('Symulowane ścieżki dwóch aktywów w czasie')
    ax.set_xlabel('Czas')
    ax.set_ylabel('Aktywo 1')
    ax.set_zlabel('Aktywo 2')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_joint_distribution(paths, asset_names=None, density_type='scatter', save=False):
    """
    Wykreśla łączną dystrybucję końcowych wartości dla dwóch aktywów.

    Parameters:
    -----------
    paths : dict
        Słownik zawierający macierze ścieżek cenowych dla każdego aktywa.
    asset_names : list, optional
        Lista nazw aktywów. Domyślnie None.
    density_type : str, optional
        Typ wykresu gęstości ('scatter', 'kde', 'hexbin'). Domyślnie 'scatter'.
    """
    if len(paths) < 2:
        raise ValueError("Potrzebne są co najmniej dwa aktywa do wizualizacji łącznego rozkładu.")

    if asset_names is None:
        asset_names = list(paths.keys())

    # Pobierz końcowe wartości dla pierwszych dwóch aktywów
    final_values_1 = paths[asset_names[0]][:, -1]
    final_values_2 = paths[asset_names[1]][:, -1]

    # Oblicz współczynnik korelacji
    correlation = np.corrcoef(final_values_1, final_values_2)[0, 1]

    # Przygotuj wykres
    plt.figure(figsize=(8, 5))

    if density_type == 'scatter':
        plt.scatter(final_values_1, final_values_2, alpha=0.5)
        plt.title(f'Łączny rozkład końcowych wartości (korelacja: {correlation:.4f})')
    elif density_type == 'kde':
        sns.kdeplot(x=final_values_1, y=final_values_2, fill=True, levels=20, cmap="Blues")
        #plt.title(f'Estymator gęstości wektora (korelacja: {correlation:.4f})')
    elif density_type == 'hexbin':
        plt.hexbin(final_values_1, final_values_2, gridsize=30, cmap='Blues')
        plt.colorbar(label='Liczba punktów')
        plt.title(f'Heksagonalny binning (korelacja: {correlation:.4f})')

    plt.xlabel('WIG20')
    plt.ylabel('KGHM')
    plt.grid(True, alpha=0.3)

    # Dodaj marginal plots
    ax1 = plt.axes([0.15, 0.85, 0.7, 0.1])
    sns.histplot(final_values_1, kde=True, ax=ax1)
    ax1.set_xlabel('')
    ax1.set_title(f'Rozkład WIG20', fontsize=8)

    ax2 = plt.axes([0.85, 0.15, 0.1, 0.7])
    sns.histplot(y=final_values_2, kde=True, ax=ax2)
    ax2.set_ylabel('')
    ax2.set_title(f'Rozkład KGHM', fontsize=8)

    plt.tight_layout()
    if save: plt.savefig('wig20vsKGHM.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_pnl(pnl, title="Rozkład końcowego P/L", save=False, cost=0):
    """
    Analizuje rozkład końcowego P/L dla wszystkich ścieżek.

    Parameters:
    -----------
    pnl: ndarray
        Macierz P/L dla wszystkich ścieżek
    title: str
        Tytuł wykresu

    Returns:
    --------
    fig: Figure
        Wykres z analizą P/L
    stats: dict
        Statystyki rozkładu P/L
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title, fontsize=10)

    # Obliczenie końcowego P/L
    final_pnl = pnl[:, -1] if pnl.ndim > 1 else pnl

    # Statystyki
    mask_nan = np.isnan(final_pnl)
    mu = np.nanmean(final_pnl)
    sigma = np.nanstd(final_pnl)
    positive_paths_pct = 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan])
    median = np.nanmedian(final_pnl)
    skew = scipy_stats.skew(final_pnl[~mask_nan])
    curtosis = scipy_stats.kurtosis(final_pnl[~mask_nan])
    liczba_nan = np.isnan(final_pnl).sum()
    cost = np.nanmean(cost)

    # Histogram z seaborn z normalizacją do procenta
    sns.histplot(final_pnl, kde=True, ax=ax, stat='percent')

    # Linie na histogramie
    ax.axvline(mu, color='r', linestyle='--', label=f"Średnia: ${mu:.2f}")
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, label="P&L = 0")

    # Dodanie statystyk do wykresu
    stats_text = (
        f"Średnia: {mu:.2f}\n"
        f"Średnia kosztów: {cost:.2f}\n"
        f"Mediana: {median:.2f}\n"
        f"Odchylenie std: {sigma:.2f}\n"
        f"Procent z zyskiem: {positive_paths_pct:.1f}%\n"
        f"Skośność: {skew:.2f}\n"
        f"Kurtoza: {curtosis:.2f}\n"
        f"Liczba NA: {liczba_nan:.2f}\n"
    )
    ax.text(0.01, 0.75, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title("Rozkład końcowego P&L")
    ax.set_xlabel('Końcowy P/L', loc='center')
    ax.set_ylabel('Procent ścieżek [%]')
    ax.legend()

    # Pozostałe statystyki
    stats = {
        'mean': mu,
        'median': median,
        'std': sigma,
        'positive_paths_pct': positive_paths_pct,
        'skewness': skew,
        'kurtosis': curtosis
    }

    plt.tight_layout()
    if save: plt.savefig('gamma2_hist_cost.png', format='png', bbox_inches='tight', dpi=300)
    return fig, stats


def plot_qq_pnl(pnl, title="Wykres kwantylowo-kwantylowy (Q-Q) P/L", save=False, cost=0):
    """
    Tworzy wykres kwantylowo-kwantylowy (Q-Q) dla rozkładu P/L w porównaniu do rozkładu normalnego.

    Parameters:
    -----------
    pnl: ndarray
        Macierz P/L dla wszystkich ścieżek (może być 1D lub 2D - w przypadku 2D brany jest ostatni okres)
    title: str
        Tytuł wykresu

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Obiekt wykresu
    stats: dict
        Statystyki opisowe rozkładu P/L
    """
    # Przygotowanie danych
    final_pnl = pnl[:, -1] if pnl.ndim > 1 else pnl

    # Obliczenie statystyk
    # Statystyki
    mask_nan = np.isnan(final_pnl)
    mu = np.nanmean(final_pnl)
    sigma = np.nanstd(final_pnl)
    positive_paths_pct = 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan])
    median = np.nanmedian(final_pnl)
    skew = scipy_stats.skew(final_pnl[~mask_nan])
    curtosis = scipy_stats.kurtosis(final_pnl[~mask_nan])
    liczba_nan = np.isnan(final_pnl).sum()
    cost = np.nanmean(cost)

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(8, 5))

    # Wykres Q-Q
    scipy_stats.probplot(final_pnl, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor('steelblue')  # Punkty danych
    ax.get_lines()[0].set_markersize(5.0)
    ax.get_lines()[1].set_color('red')  # Linia referencyjna
    ax.get_lines()[1].set_linewidth(2.0)

    # Dodanie statystyk do wykresu
    stats_text = (
        f"Średnia: {mu:.2f}\n"
        f"Średnia kosztów: {cost:.2f}\n"
        f"Mediana: {median:.2f}\n"
        f"Odchylenie std: {sigma:.2f}\n"
        f"Procent z zyskiem: {positive_paths_pct:.1f}%\n"
        f"Skośność: {skew:.2f}\n"
        f"Kurtoza: {curtosis:.2f}\n"
        f"Liczba NA: {liczba_nan:.2f}\n"
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Formatowanie wykresu
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Kwantyle teoretyczne (rozkład normalny)')
    ax.set_ylabel('Kwantyle empiryczne (P/L)')
    ax.grid(True, alpha=0.3)

    # Statystyki do zwrócenia
    stats = {
        'mean': mu,
        'std': sigma,
        'skewness': skew,
        'curtosis': curtosis
    }
    plt.tight_layout()
    if save: plt.savefig('gamma2_qq_cost.png', format='png', bbox_inches='tight', dpi=300)
    return fig, stats


def plot_extreme_paths_separately(portfolio_composition, pnl, n_extreme=10,
                                 title="Ekstremalne ścieżki - zyski i straty", save=False):
    """
    Pokazuje osobne wykresy dla n ścieżek z największym zyskiem i n ścieżek z największą stratą.
    Pary (portfolio, opcja) dla tej samej ścieżki mają ten sam kolor.

    Parameters:
    -----------
    portfolio_composition: dict
        Słownik zwracany przez funkcję wallet_delta, zawierający:
        - 'portfolio_value': wartość portfela w czasie
        - 'option_value': wartość opcji w czasie
    pnl: ndarray
        Macierz P/L (używana do określenia których ścieżek użyć)
    n_extreme: int, optional (domyślnie 10)
        Liczba ścieżek do pokazania w każdej kategorii
    title: str, optional
        Tytuł główny wykresu

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Obiekt wykresu zawierający dwa podwykresy
    """
    # Pobranie danych
    portfolio = portfolio_composition['portfolio_value']
    option = portfolio_composition['option_value']
    time_points = portfolio.shape[1]
    time_axis = np.linspace(0, 1, time_points)

    # Identyfikacja ekstremalnych ścieżek
    final_pnl = pnl[:, -1] if pnl.ndim > 1 else pnl
    sorted_indices = np.argsort(final_pnl)

    # Największe straty i zyski
    worst_indices = sorted_indices[:n_extreme]
    best_indices = sorted_indices[-n_extreme:]

    # Generowanie unikalnych kolorów dla każdej ścieżki
    colors = plt.cm.Set1(np.linspace(0, 1, n_extreme))

    # Przygotowanie wykresu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle(title, y=1.05, fontsize=14)

    # Wykres dla największych strat
    for idx, i in enumerate(worst_indices):
        color = colors[idx]
        ax1.plot(time_axis, portfolio[i], color='red', alpha=0.8, linewidth=1,
                 label=f'Portfel {idx+1}' if idx == 0 else "")
        ax1.plot(time_axis, option[i], color='blue', alpha=0.8, linewidth=1, linestyle='-',
                 label=f'Opcja {idx+1}' if idx == 0 else "")

    ax1.set_title(f'Największa strata')
    ax1.set_xlabel('Czas')
    ax1.set_ylabel('Wartość')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Wykres dla największych zysków
    for idx, i in enumerate(best_indices):
        color = colors[idx]
        ax2.plot(time_axis, portfolio[i], color='red', alpha=0.8, linewidth=1,
                 label=f'Portfel {idx+1}' if idx == 0 else "")
        ax2.plot(time_axis, option[i], color='blue', alpha=0.8, linewidth=1, linestyle='-',
                 label=f'Opcja {idx+1}' if idx == 0 else "")

    ax2.set_title(f'Największy zysk')
    ax2.set_xlabel('Czas')
    ax2.set_ylabel('Wartość')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Dodanie informacji o wartościach P/L
    min_pnl = np.min(final_pnl[worst_indices])
    max_pnl = np.max(final_pnl[best_indices])
    fig.text(0.5, 0.93,
             f"Zakres P/L: od {min_pnl:.2f} (największa strata) do {max_pnl:.2f} (największy zysk)",
             ha='center', fontsize=10)

    plt.tight_layout()
    if save: plt.savefig('extreme_paths.png', format='png', bbox_inches='tight', dpi=300)
    return fig

### WALLET
def perform_hedging_frequency_test(paths_gbm, params, K, option_type='call', save=False, wallet='delta'):
    """
    Przeprowadza test wpływu częstotliwości rehedgingu na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    K: float
        Cena wykonania opcji
    option_type: str
        Typ opcji ('call' lub 'put')

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej częstotliwości
    """


    # Konfiguracja testów
    frequencies = {
        #'Co 2 tygodnie': 26,
        #'Tygodniowo': 52,
        #'Co 3 dni': 252 // 3,
        'Dziennie': 252,
        '2x dziennie': 252 * 2,
        '5x dziennie': 252 * 5
    }

    # Przygotowanie wyników
    results = []
    figs = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(len(frequencies), 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu częstotliwości rehedgingu na rozkład P&L', y=1.02, fontsize=14)

    for idx, (freq_name, n_hedge) in enumerate(frequencies.items()):

        if wallet == 'delta':
            # Obliczenie P&L
            pnl, _ = wallet_delta(
                paths=paths_gbm,
                T_years=1,
                sigma=params['volatility'],
                r=params['r'],
                K=K,
                option_type=option_type,
                n_hedge=n_hedge
            )
        elif wallet == 'gamma':
            K_binary = K
            pnl, _ = wallet_delta_gamma(paths_gbm,
                                        1,
                                        params['volatility'],
                                        params['r'],
                                        K, K_binary,
                                        option_type='call',
                                        n_hedge=n_hedge,
                                        binary_option_type='binary_call')

        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        mask_nan = np.isnan(final_pnl)
        stats_data = {
            'Częstotliwość': freq_name,
            'Liczba rehedge': n_hedge,
            'Średnia': np.nanmean(final_pnl),
            'Mediana': np.nanmedian(final_pnl),
            'Std': np.nanstd(final_pnl),
            'Skewness':scipy_stats.skew(final_pnl[~mask_nan]),
            'Kurtosis': scipy_stats.kurtosis(final_pnl[~mask_nan]),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Wykres histogramu
        ax1 = axes[idx, 0] if len(frequencies) > 1 else axes[0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'{freq_name} (n={n_hedge}) - Histogram', fontsize=10)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_ylim(0, 7)
        ax1.set_xlim(-60, 60)
        ax1.set_xlabel('')
        ax1.set_ylabel('')

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.8)
                 )

        # Wykres Q-Q
        ax2 = axes[idx, 1] if len(frequencies) > 1 else axes[1]
        scipy_stats.probplot(final_pnl, dist="norm", plot=ax2)
        ax2.set_title(f'{freq_name} (n={n_hedge}) - Wykres Q-Q', fontsize=10)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_ylim(-70, 70)
        ax2.set_xlabel('')
        ax2.set_ylabel('')

    axes[2, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[2, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)

    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)
    #plt.tight_layout()

    if save: plt.savefig('delta_rehedge_2.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[[
        'Częstotliwość', 'Liczba rehedge', 'Średnia', 'Mediana', 'Std',
        'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)'
    ]]

    return fig, stats_df


def compare_pnl_by_strike(paths_gbm, params, strikes=None, option_type='call', save=False):
    """
    Analiza wpływu wartości Strike na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    strikes: list
        Lista cen wykonania opcji
    option_type: str
        Typ opcji ('call' lub 'put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości Strike
    """

    if strikes is None or len(strikes) > 3:
        strikes = [1700, 1800, 1900]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu wartości Strike na rozkład P&L', y=1.02, fontsize=14)

    for idx, strike in enumerate(strikes):
        # Obliczenie P&L
        pnl, _ = wallet_delta(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=strike,
            option_type=option_type,
            n_hedge=252
        )
        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        stats_data = {
            'Strike': strike,
            'Średnia': np.mean(final_pnl),
            'Mediana': np.median(final_pnl),
            'Std': np.std(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl),
            'Kurtosis': scipy_stats.kurtosis(final_pnl),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Histogram
        ax1 = axes[idx, 0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'Strike {strike} - Histogram', fontsize=12)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_xlabel('')
        ax1.set_ylabel('')

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.8))

        ax1.set_xlim(-60, 60)
        ax1.set_ylim(0, 6)

        # Wykres Q-Q
        ax2 = axes[idx, 1]
        scipy_stats.probplot(final_pnl, dist="norm", plot=ax2)
        ax2.set_title(f'Strike {strike} - Wykres Q-Q', fontsize=12)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_ylim(-90, 90)

    # Etykiety osi
    axes[-1, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[-1, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)
    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)

    #plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig('compare_pnl_strikes_2.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['Strike', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df

def compare_pnl_by_volatility(paths_gbm, params, K, volatility=None, option_type='call', save=False):
    """
    Analiza wpływu wartości volatility na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    K: int
        Strike
    volatility: list
        Lista volatilites
    option_type: str
        Typ opcji ('call' lub 'put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości Strike
    """

    if volatility is None or len(volatility) != 3:
        volatility = [params['volatility']*1.1, params['volatility'], 0.9*params['volatility']]

    vol_ratio = (np.array(volatility) / params['volatility']) * 100

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu wartości Volatiltiy na rozkład P&L', y=1.02, fontsize=14)

    for idx, volatility in enumerate(volatility):
        # Obliczenie P&L
        pnl, _ = wallet_delta(
            paths=paths_gbm,
            T_years=1,
            sigma=volatility,
            r=params['r'],
            K=K,
            option_type=option_type,
            n_hedge=252
        )
        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        stats_data = {
            'Strike': K,
            'Średnia': np.mean(final_pnl),
            'Mediana': np.median(final_pnl),
            'Std': np.std(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl),
            'Kurtosis': scipy_stats.kurtosis(final_pnl),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Histogram
        ax1 = axes[idx, 0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'Założone {(volatility/params['volatility']) * 100:.2f}% zrealizowanego volatility - Histogram', fontsize=9)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xlim(-60, 100)
        ax1.set_ylim(0, 7)

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8))

        # Wykres Q-Q
        ax2 = axes[idx, 1]
        scipy_stats.probplot(final_pnl, dist="norm", plot=ax2)
        ax2.set_title(f'Założone {(volatility/params['volatility']) * 100:.2f}% zrealizowanego volatility - Wykres Q-Q', fontsize=9)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        #ax1.set_xlim(-120, 50)
        ax2.set_ylim(-70, 120)

    # Etykiety osi
    axes[-1, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[-1, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)
    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)

    #plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save:
        plt.savefig('compare_pnl_vol_2.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['Strike', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df


def compare_pnl_by_mu(N, params, K, mu=None, option_type='call', save=False):
    """
    Analiza wpływu wartości volatility na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    K: int
        Strike
    volatility: list
        Lista volatilites
    option_type: str
        Typ opcji ('call' lub 'put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości Strike
    """

    if mu is None or len(mu) != 3:
        mu = [params['mu']*1.1, params['mu'], 0.9*params['mu']]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu wartości Mu na rozkład P&L', y=1.02, fontsize=14)

    for idx, mu in enumerate(mu):
        paths_gbm, _ = generate_gbm_paths(
            N=N,
            T=1,
            s0=params['s0'],
            mean=mu,
            volatility=params['volatility'],
            h=1,
            plot=False
        )

        # Obliczenie P&L
        pnl, _ = wallet_delta(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            option_type=option_type,
            n_hedge=252
        )
        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        stats_data = {
            'Strike': K,
            'Średnia': np.mean(final_pnl),
            'Mediana': np.median(final_pnl),
            'Std': np.std(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl),
            'Kurtosis': scipy_stats.kurtosis(final_pnl),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Histogram
        ax1 = axes[idx, 0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'Założone {(mu/params['mu']) * 100:.2f}% zrealizowanej średniej - Histogram', fontsize=9)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xlim(-60, 70)
        ax1.set_ylim(0, 6)

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8))

        # Wykres Q-Q
        ax2 = axes[idx, 1]
        scipy_stats.probplot(final_pnl, dist="norm", plot=ax2)
        ax2.set_title(f'Założone {(mu/params['mu']) * 100:.2f}% zrealizowanej średniej - Wykres Q-Q', fontsize=9)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        #ax1.set_xlim(-120, 50)
        ax2.set_ylim(-70, 70)

    # Etykiety osi
    axes[-1, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[-1, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)
    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)

    #plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save:
        plt.savefig('compare_pnl_mu_1.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['Strike', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df

def compare_pnl_by_r(paths_gbm, params, K, r=None, option_type='call', save=False):
    """
    Analiza wpływu wartości volatility na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    K: int
        Strike
    volatility: list
        Lista volatilites
    option_type: str
        Typ opcji ('call' lub 'put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości Strike
    """

    if r is None or len(r) != 3:
        r = [params['r']*1.1, params['r'], 0.9*params['r']]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu stopy zwrotu na rozkład P&L', y=1.02, fontsize=14)

    for idx, r in enumerate(r):
        # Obliczenie P&L
        pnl, _ = wallet_delta(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=r,
            K=K,
            option_type=option_type,
            n_hedge=252
        )
        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        stats_data = {
            'Strike': K,
            'Średnia': np.mean(final_pnl),
            'Mediana': np.median(final_pnl),
            'Std': np.std(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl),
            'Kurtosis': scipy_stats.kurtosis(final_pnl),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Histogram
        ax1 = axes[idx, 0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'Założone {(r/params['r']) * 100:.2f}% r - Histogram', fontsize=9)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        #ax1.set_xlim(-60, 100)
        #ax1.set_ylim(0, 7)

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8))

        # Wykres Q-Q
        ax2 = axes[idx, 1]
        scipy_stats.probplot(final_pnl, dist="norm", plot=ax2)
        ax2.set_title(f'Założone {(r/params['r']) * 100:.2f}% zrealizowanego volatility - Wykres Q-Q', fontsize=9)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        #ax1.set_xlim(-120, 50)
        #ax2.set_ylim(-70, 120)

    # Etykiety osi
    axes[-1, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[-1, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)
    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)

    #plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save:
        plt.savefig('compare_pnl_r_1.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['Strike', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df

def plot_pnl_by_K(paths_gbm, params, n_hedge=252, K_values=None, save=False):

    if K_values is None:
        K_values = [1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
    results = np.zeros((len(K_values), 5))  # 5 kolumn: K, q25, q50, q75, mean

    for i, K in enumerate(K_values):
        pnl, _ = wallet_delta(paths=paths_gbm, T_years=1, sigma=params['volatility'] * 1.01, r=params['r'], K=K,
                              option_type='call', n_hedge=n_hedge)

        pnl_final = pnl

        # Zapisz K i statystyki
        results[i, 0] = K
        results[i, 1] = np.percentile(pnl_final, 25)  # q25
        results[i, 2] = np.percentile(pnl_final, 50)  # mediana (q50)
        results[i, 3] = np.percentile(pnl_final, 75)  # q75
        results[i, 4] = np.mean(pnl_final)  # średnia

    # Tworzenie wykresu
    plt.figure(figsize=(12, 7))

    # Rysowanie linii dla każdej statystyki
    plt.plot(results[:, 0], results[:, 1], 'b-', marker='o', label='Kwantyl 25%')
    plt.plot(results[:, 0], results[:, 2], 'g-', marker='s', label='Mediana')
    plt.plot(results[:, 0], results[:, 3], 'r-', marker='^', label='Kwantyl 75%')
    plt.plot(results[:, 0], results[:, 4], 'k--', marker='*', label='Średnia')

    # Dodawanie opisów
    plt.xlabel('Wartość K')
    plt.ylabel('PNL')
    plt.title('Statystyki PNL w zależności od K')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Wyświetlenie wykresu
    plt.tight_layout()

    if save:
        plt.savefig('pnl_by_K.png', format='png', bbox_inches='tight', dpi=300)

    plt.show()

def delta_rehedge_margin_analysis(paths_gbm, params, rehedge_margins=None, K=1700, option_type='call',
                            save=False):
    """
    Analiza wpływu parametru rehedge_margin na rozkład P&L w strategii delta-gamma hedgingu.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    rehedge_margins: list
        Lista wartości parametru rehedge_margin do analizy
    K: float
        Cena wykonania dla opcji waniliowej
    option_type: str
        Typ opcji ('call' lub 'put')
    binary_option_type: str
        Typ opcji binarnej ('binary_call' lub 'binary_put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości rehedge_margin
    """

    if rehedge_margins is None or len(rehedge_margins) != 6:
        # Domyślne wartości rehedge_margin
        rehedge_margins = [0.1, 0.2, 0.3, 0.5, 0.7, 1]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle('Analiza wpływu parametru rehedge_margin na rozkład P&L (Delta Hedging)',
                 y=1.02, fontsize=14)

    # Spłaszczenie tablicy axes dla łatwiejszego iterowania
    axes_flat = axes.flatten()

    for idx, margin in enumerate(rehedge_margins):
        # Obliczenie P&L przy użyciu strategii delta-gamma
        pnl, portfolio_composition = wallet_delta_2(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            option_type=option_type,
            n_hedge=52,
            rehedge_margin=margin,
            delta_margin=0.1,  # Stała wartość
        )

        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        mask_nan = np.isnan(final_pnl)
        stats_data = {
            'Rehedge Margin': margin,
            'Średnia': np.nanmean(final_pnl),
            'Średnia kosztów': np.nanmean(portfolio_composition['cost']),
            'Mediana': np.nanmedian(final_pnl),
            'Std': np.nanstd(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl[~mask_nan]),
            'Kurtosis': scipy_stats.kurtosis(final_pnl[~mask_nan]),
            'Min': np.min(final_pnl[~mask_nan]),
            'Max': np.max(final_pnl[~mask_nan]),
            'P(PL>0)': 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan]),
            'NA': np.isnan(final_pnl).sum()
        }
        results.append(stats_data)

        # Histogram P&L dla bieżącego rehedge_margin
        ax = axes_flat[idx]
        sns.histplot(final_pnl, kde=True, ax=ax, stat='percent', color='blue', bins=30)
        ax.set_title(f'rehedge_margin = {margin} - Histogram P&L', fontsize=12)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)

        # Dodanie linii średniej i mediany
        ax.axvline(stats_data['Średnia'], color='green', linestyle=':', linewidth=1, label='Średnia')
        ax.axvline(stats_data['Mediana'], color='purple', linestyle=':', linewidth=1, label='Mediana')

        # Dodanie statystyk do wykresu
        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Średnia kosztów: {stats_data['Średnia kosztów']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}\n")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8))

        # Ustawienie etykiet osi
        ax.set_xlabel('Końcowy P/L' if idx >= 4 else '')
        ax.set_ylabel('Procent ścieżek [%]' if idx % 2 == 0 else '')

        # Dodanie legendy tylko do pierwszego wykresu
        if idx == 0:
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig('delta_margin_analysis.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[
        ['Rehedge Margin', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df


##### wallet_delta_gamma


def gamma_bstrike(paths_gbm, params, binary_strikes=None, K=1800, option_type='call', binary_option_type='binary_call',
                  save=False):
    """
    Analiza wpływu wartości Strike opcji binarnej na rozkład P&L w strategii delta-gamma hedgingu.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    binary_strikes: list
        Lista cen wykonania opcji binarnych
    K: float
        Cena wykonania dla opcji waniliowej
    option_type: str
        Typ opcji ('call' lub 'put')
    binary_option_type: str
        Typ opcji binarnej ('binary_call' lub 'binary_put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości Strike opcji binarnej
    """

    if binary_strikes is None or len(binary_strikes) != 6:
        # Domyślne wartości strajków dla opcji binarnej - wokół strajku opcji głównej
        binary_strikes = [K - 300, K - 200, K - 100, K, K + 100, K + 200]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu wartości Strike opcji binarnej na rozkład P&L (Delta-Gamma Hedging)',
                 y=1.02, fontsize=14)

    # Spłaszczenie tablicy axes dla łatwiejszego iterowania
    axes_flat = axes.flatten()

    for idx, binary_strike in enumerate(binary_strikes):
        # Obliczenie P&L przy użyciu strategii delta-gamma
        pnl, portfolio_composition = wallet_delta_gamma(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            K_binary=binary_strike,
            option_type=option_type,
            binary_option_type=binary_option_type,
            n_hedge=52
        )

        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk

        mask_nan = np.isnan(final_pnl)
        stats_data = {
            'Binary Strike': binary_strike,
            'Średnia': np.nanmean(final_pnl),
            'Mediana': np.nanmean(final_pnl),
            'Std': np.nanstd(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl[~mask_nan]),
            'Kurtosis': scipy_stats.kurtosis(final_pnl[~mask_nan]),
            'Min': np.min(final_pnl[~mask_nan]),
            'Max': np.max(final_pnl[~mask_nan]),
            'P(PL>0)': 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan]),
            'NA': np.isnan(final_pnl).sum(),
            'cost': np.nanmean(portfolio_composition['cost'])
        }
        results.append(stats_data)

        # Histogram P&L dla bieżącego binary strike
        ax = axes_flat[idx]
        sns.histplot(final_pnl, kde=True, ax=ax, stat='percent', color='blue')
        ax.set_title(f'Binary Strike {binary_strike} - Histogram P&L', fontsize=12)
        ax.axvline(0, color='red', linestyle='--')

        # Dodanie statystyk do wykresu
        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Średnia kosztów: {stats_data['cost']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Liczba NA: {stats_data['NA']}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8))

        # Ustawienie etykiet osi dla wszystkich wykresów
        ax.set_xlabel('Końcowy P/L' if idx >= 4 else '')
        ax.set_ylabel('Procent ścieżek [%]' if idx % 2 == 0 else '')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig('gamma_bstrike_analysis.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['Binary Strike', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df


def gamma_rehedge_margin_analysis(paths_gbm, params, rehedge_margins=None, K=1700, option_type='call',
                            binary_option_type='binary_call',
                            save=False):
    """
    Analiza wpływu parametru rehedge_margin na rozkład P&L w strategii delta-gamma hedgingu.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    rehedge_margins: list
        Lista wartości parametru rehedge_margin do analizy
    K: float
        Cena wykonania dla opcji waniliowej
    option_type: str
        Typ opcji ('call' lub 'put')
    binary_option_type: str
        Typ opcji binarnej ('binary_call' lub 'binary_put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości rehedge_margin
    """

    if rehedge_margins is None or len(rehedge_margins) != 6:
        # Domyślne wartości rehedge_margin
        rehedge_margins = [0.1, 0.5, 0.7, 0.9, 1.2, 1.5]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle('Analiza wpływu parametru rehedge_margin na rozkład P&L (Delta-Gamma Hedging z opcją binarną Call)',
                 y=1.02, fontsize=14)

    # Spłaszczenie tablicy axes dla łatwiejszego iterowania
    axes_flat = axes.flatten()

    for idx, margin in enumerate(rehedge_margins):
        # Obliczenie P&L przy użyciu strategii delta-gamma
        pnl, portfolio_composition = wallet_delta_gamma(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            K_binary=1100,  # Używamy tego samego strike'a dla opcji binarnej
            option_type=option_type,
            binary_option_type=binary_option_type,
            n_hedge=52,
            rehedge_margin=margin,
            delta_margin=0.1,  # Stała wartość
            binary_gamma_margin=1e-3  # Stała wartość
        )

        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        mask_nan = np.isnan(final_pnl)
        stats_data = {
            'Rehedge Margin': margin,
            'Średnia': np.nanmean(final_pnl),
            'Średnia kosztów': np.nanmean(portfolio_composition['cost']),
            'Mediana': np.nanmedian(final_pnl),
            'Std': np.nanstd(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl[~mask_nan]),
            'Kurtosis': scipy_stats.kurtosis(final_pnl[~mask_nan]),
            'Min': np.min(final_pnl[~mask_nan]),
            'Max': np.max(final_pnl[~mask_nan]),
            'P(PL>0)': 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan]),
            'NA': np.isnan(final_pnl).sum()
        }
        results.append(stats_data)

        # Histogram P&L dla bieżącego rehedge_margin
        ax = axes_flat[idx]
        sns.histplot(final_pnl, kde=True, ax=ax, stat='percent', color='blue', bins=30)
        ax.set_title(f'rehedge_margin = {margin} - Histogram P&L', fontsize=12)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)

        # Dodanie linii średniej i mediany
        ax.axvline(stats_data['Średnia'], color='green', linestyle=':', linewidth=1, label='Średnia')
        ax.axvline(stats_data['Mediana'], color='purple', linestyle=':', linewidth=1, label='Mediana')

        # Dodanie statystyk do wykresu
        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Koszty: {stats_data['Średnia kosztów']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}\n")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8))

        # Ustawienie etykiet osi
        ax.set_xlabel('Końcowy P/L' if idx >= 4 else '')
        ax.set_ylabel('Procent ścieżek [%]' if idx % 2 == 0 else '')

        # Dodanie legendy tylko do pierwszego wykresu
        if idx == 0:
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig('gamma_rehedge_margin_analysis_call.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[
        ['Rehedge Margin', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df

def compare_hedge_strategies(paths_gbm, params, K=1700, binary_strikes=[1600, 1700, 1800, 1900, 2000],
                             option_type='call', binary_option_type='binary_call', save=False):
    """
    Funkcja porównująca zwroty strategii delta-hedge i delta-gamma-hedge dla różnych wartości
    strike opcji binarnej w postaci wykresu liniowego średnich zwrotów.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    K: float
        Cena wykonania dla opcji waniliowej
    binary_strikes: list
        Lista wartości strike dla opcji binarnych w strategii delta-gamma
    option_type: str
        Typ opcji ('call' lub 'put')
    binary_option_type: str
        Typ opcji binarnej ('binary_call' lub 'binary_put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykres porównawczy
    results_df: DataFrame
        Ramka danych z wynikami
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Zwiększenie odstępu między wykresami
    plt.subplots_adjust(hspace=0.3)

    # Jednolity styl dla wszystkich wykresów
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Główny tytuł z odpowiednim rozmiarem i odstępem
    fig.suptitle('Porównanie strategii hedgingu: Delta vs Delta-Gamma',
                 fontsize=16, fontweight='bold', y=0.98)

    # Wartości parametrów do analizy
    n_hedge_values = np.array([12, 26, 52, 104, 252])  # Liczba rehedgingów w roku
    volatility_values = np.array([0.66, 0.75, 0.9, 1, 1.1, 1.25, 1.33]) * params['volatility']  # Zmienności roczne
    strike_values = np.array([K* 0.7, K * 0.8, K * 0.9, K, K * 1.1, K * 1.2, K * 1.3])  # Wartości strike dla opcji waniliowej

    # Kolory dla różnych opcji binarnych - użycie bardziej wyraźnej palety
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(binary_strikes)))

    # 1. Analiza w zależności od liczby rehedgingów
    ax = axes[0]

    # Przygotowanie wyników
    results_rehedge = []

    # Zwykły delta hedge
    for n_hedge in n_hedge_values:
        pnl, _ = wallet_delta_2(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            option_type=option_type,
            n_hedge=int(n_hedge)
        )

        results_rehedge.append({
            'Strategia': 'Delta Hedge',
            'Parametr': n_hedge,
            'Średni PL': np.mean(pnl),
            'Std PL': np.std(pnl),
            'Binary Strike': None
        })

    # Delta-gamma hedge dla różnych binary strikes
    for i, binary_strike in enumerate(binary_strikes):
        for n_hedge in n_hedge_values:
            pnl, _ = wallet_delta_gamma(
                paths=paths_gbm,
                T_years=1,
                sigma=params['volatility'],
                r=params['r'],
                K=K,
                K_binary=binary_strike,
                option_type=option_type,
                binary_option_type=binary_option_type,
                n_hedge=int(n_hedge)
            )

            # Usuwanie wartości NA przed obliczeniem średniej i odchylenia standardowego
            valid_pnl = pnl[~np.isnan(pnl)]

            results_rehedge.append({
                'Strategia': f'Delta-Gamma Hedge (K_bin={binary_strike})',
                'Parametr': n_hedge,
                'Średni PL': np.mean(valid_pnl),
                'Std PL': np.std(valid_pnl),
                'Binary Strike': binary_strike
            })

    # Konwersja do DataFrame i wizualizacja
    df_rehedge = pd.DataFrame(results_rehedge)

    # Rysowanie linii dla Delta Hedge
    delta_data = df_rehedge[df_rehedge['Strategia'] == 'Delta Hedge']
    ax.plot(delta_data['Parametr'], delta_data['Średni PL'], 'o-', color='black', linewidth=2.5,
            markersize=8, label='Delta Hedge')

    # Rysowanie linii dla Delta-Gamma Hedge z różnymi binary strikes
    for i, binary_strike in enumerate(binary_strikes):
        gamma_data = df_rehedge[df_rehedge['Binary Strike'] == binary_strike]
        ax.plot(gamma_data['Parametr'], gamma_data['Średni PL'], 'o-', color=colors[i], linewidth=2,
                markersize=6, label=f'Delta-Gamma (K_bin={binary_strike})')

    ax.set_title('Wpływ liczby rehedgingów na średni P&L', fontweight='bold')
    ax.set_xlabel('Liczba rehedgingów w roku')
    ax.set_ylabel('Średni P&L')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=8,
          labelspacing=0.2, handlelength=1, handletextpad=0.4)


    # 2. Analiza w zależności od zmienności
    ax = axes[1]

    # Przygotowanie wyników
    results_vol = []

    # Zwykły delta hedge
    for vol in volatility_values:
        # Tutaj można zaktualizować ścieżki lub użyć nowych ścieżek z odpowiednią volatility
        pnl, _ = wallet_delta_2(
            paths=paths_gbm,  # Tutaj powinna być ścieżka z odpowiednią volatility
            T_years=1,
            sigma=vol,
            r=params['r'],
            K=K,
            option_type=option_type,
            n_hedge=52
        )

        results_vol.append({
            'Strategia': 'Delta Hedge',
            'Parametr': vol,
            'Średni PL': np.mean(pnl),
            'Std PL': np.std(pnl),
            'Binary Strike': None
        })

    # Delta-gamma hedge dla różnych binary strikes
    for i, binary_strike in enumerate(binary_strikes):
        for vol in volatility_values:
            pnl, _ = wallet_delta_gamma(
                paths=paths_gbm,  # Tutaj powinna być ścieżka z odpowiednią volatility
                T_years=1,
                sigma=vol,
                r=params['r'],
                K=K,
                K_binary=binary_strike,
                option_type=option_type,
                binary_option_type=binary_option_type,
                n_hedge=52
            )

            # Usuwanie wartości NA przed obliczeniem średniej i odchylenia standardowego
            valid_pnl = pnl[~np.isnan(pnl)]

            results_vol.append({
                'Strategia': f'Delta-Gamma Hedge (K_bin={binary_strike})',
                'Parametr': vol,
                'Średni PL': np.mean(valid_pnl),
                'Std PL': np.std(valid_pnl),
                'Binary Strike': binary_strike
            })

    # Konwersja do DataFrame i wizualizacja
    df_vol = pd.DataFrame(results_vol)

    # Rysowanie linii dla Delta Hedge
    delta_data = df_vol[df_vol['Strategia'] == 'Delta Hedge']
    ax.plot(delta_data['Parametr'], delta_data['Średni PL'], 'o-', color='black', linewidth=2.5,
            markersize=8, label='Delta Hedge')

    # Rysowanie linii dla Delta-Gamma Hedge z różnymi binary strikes
    for i, binary_strike in enumerate(binary_strikes):
        gamma_data = df_vol[df_vol['Binary Strike'] == binary_strike]
        ax.plot(gamma_data['Parametr'], gamma_data['Średni PL'], 'o-', color=colors[i], linewidth=2,
                markersize=6, label=f'Delta-Gamma (K_bin={binary_strike})')

    ax.set_title('Wpływ zmienności na średni P&L', fontweight='bold')
    ax.set_xlabel('Zmienność roczna')
    ax.set_ylabel('Średni P&L')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=8,
          labelspacing=0.2, handlelength=1, handletextpad=0.4)

    # 3. Analiza w zależności od strike opcji waniliowej
    ax = axes[2]

    # Przygotowanie wyników
    results_strike = []

    # Zwykły delta hedge
    for strike in strike_values:
        pnl, _ = wallet_delta_2(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=strike,
            option_type=option_type,
            n_hedge=252
        )

        results_strike.append({
            'Strategia': 'Delta Hedge',
            'Parametr': strike,
            'Średni PL': np.mean(pnl),
            'Std PL': np.std(pnl),
            'Binary Strike': None
        })

    # Delta-gamma hedge dla różnych binary strikes
    for i, binary_strike in enumerate(binary_strikes):
        for strike in strike_values:
            # Dostosowanie binary_strike względem strike
            relative_binary_strike = binary_strike * (strike / K)

            pnl, _ = wallet_delta_gamma(
                paths=paths_gbm,
                T_years=1,
                sigma=params['volatility'],
                r=params['r'],
                K=strike,
                K_binary=relative_binary_strike,
                option_type=option_type,
                binary_option_type=binary_option_type,
                n_hedge=252
            )

            # Usuwanie wartości NA przed obliczeniem średniej i odchylenia standardowego
            valid_pnl = pnl[~np.isnan(pnl)]

            results_strike.append({
                'Strategia': f'Delta-Gamma Hedge (K_bin={binary_strike})',
                'Parametr': strike,
                'Średni PL': np.mean(valid_pnl),
                'Std PL': np.std(valid_pnl),
                'Binary Strike': binary_strike,
                'Relative Binary Strike': relative_binary_strike
            })

    # Konwersja do DataFrame i wizualizacja
    df_strike = pd.DataFrame(results_strike)

    # Rysowanie linii dla Delta Hedge
    delta_data = df_strike[df_strike['Strategia'] == 'Delta Hedge']
    ax.plot(delta_data['Parametr'], delta_data['Średni PL'], 'o-', color='black', linewidth=2.5,
            markersize=8, label='Delta Hedge')

    # Rysowanie linii dla Delta-Gamma Hedge z różnymi binary strikes
    for i, binary_strike in enumerate(binary_strikes):
        gamma_data = df_strike[df_strike['Binary Strike'] == binary_strike]
        ax.plot(gamma_data['Parametr'], gamma_data['Średni PL'], 'o-', color=colors[i], linewidth=2,
                markersize=6, label=f'Delta-Gamma (K_bin={binary_strike})')

    ax.set_title('Wpływ ceny wykonania opcji waniliowej na średni P&L', fontweight='bold')
    ax.set_xlabel('Strike opcji waniliowej')
    ax.set_ylabel('Średni P&L')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=8,
          labelspacing=0.2, handlelength=1, handletextpad=0.4)

    # Lepsze dopasowanie układu wykresu
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig('compare_hedge_strategies.png', format='png', bbox_inches='tight', dpi=300)

    # Łączenie wszystkich wyników
    results_df = pd.concat([
        df_rehedge.assign(Analiza='Liczba rehedgingów'),
        df_vol.assign(Analiza='Zmienność'),
        df_strike.assign(Analiza='Strike opcji waniliowej')
    ])

    return fig, results_df


def plot_deltas(paths, title, label, save, model):
    label = label
    color = 'red'
    quantiles_paths = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    time = np.arange(paths.shape[1])

    # Oblicz min i max dla każdego punktu czasowego
    min_per_time = np.min(paths, axis=0)  # Minimum dla każdego t
    max_per_time = np.max(paths, axis=0)  # Maximum dla każdego t

    fig, axes = plt.subplots(figsize=(8, 5))

    fig.suptitle(title)
    # Mediana
    plt.plot(time, quantiles_paths[2], label=f"Mediana {label}", color=color)

    # Przedziały kwantylowe
    plt.fill_between(time, quantiles_paths[1], quantiles_paths[3], color=color, alpha=0.2,
                     label=f"IQR (25%-75%) {label}")

    # Ekstremalne kwantyle
    plt.plot(time, quantiles_paths[0], color=color, ls=':', label=f'5% / 95% {label}')
    plt.plot(time, quantiles_paths[4], color=color, ls=':')

    # Dodane linie min i max
    plt.plot(time, min_per_time, color='blue', ls='--', alpha=0.7, label='Minimum')
    plt.plot(time, max_per_time, color='green', ls='--', alpha=0.7, label='Maximum')

    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f'{label}_{model}_plot.png', format='png', bbox_inches='tight', dpi=300)


    plt.show()


def gamma2_rehedge_margin_analysis(paths_gbm, params, rehedge_margins=None, K=1700, option_type='call',
                            save=False):
    """
    Analiza wpływu parametru rehedge_margin na rozkład P&L w strategii delta-gamma hedgingu.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    rehedge_margins: list
        Lista wartości parametru rehedge_margin do analizy
    K: float
        Cena wykonania dla opcji waniliowej
    option_type: str
        Typ opcji ('call' lub 'put')
    binary_option_type: str
        Typ opcji binarnej ('binary_call' lub 'binary_put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości rehedge_margin
    """

    if rehedge_margins is None or len(rehedge_margins) != 6:
        # Domyślne wartości rehedge_margin
        rehedge_margins = [3, 4, 5, 6, 7, 8]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle('Analiza wpływu parametru rehedge_margin na rozkład P&L (Delta-Gamma Hedging z statycznymi dwiema opcjami binarnymi)',
                 y=1.02, fontsize=14)

    # Spłaszczenie tablicy axes dla łatwiejszego iterowania
    axes_flat = axes.flatten()

    for idx, margin in enumerate(rehedge_margins):
        # Obliczenie P&L przy użyciu strategii delta-gamma
        pnl, portfolio_composition = wallet_delta_gamma_2(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            K_binaries=[],
            option_type=option_type,
            n_hedge=52,
            rehedge_margin=margin,
            delta_margin=0.1,  # Stała wartość
            denom_margin=1e-3  # Stała wartość
        )

        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        mask_nan = np.isnan(final_pnl)
        stats_data = {
            'Rehedge Margin': margin,
            'Średnia': np.nanmean(final_pnl),
            'Średnia kosztów': np.nanmean(portfolio_composition['cost']),
            'Mediana': np.nanmedian(final_pnl),
            'Std': np.nanstd(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl[~mask_nan]),
            'Kurtosis': scipy_stats.kurtosis(final_pnl[~mask_nan]),
            'Min': np.min(final_pnl[~mask_nan]),
            'Max': np.max(final_pnl[~mask_nan]),
            'P(PL>0)': 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan]),
            'NA': np.isnan(final_pnl).sum()
        }
        results.append(stats_data)

        # Histogram P&L dla bieżącego rehedge_margin
        ax = axes_flat[idx]
        sns.histplot(final_pnl, kde=True, ax=ax, stat='percent', color='blue', bins=30)
        ax.set_title(f'rehedge_margin = {margin} - Histogram P&L', fontsize=12)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)

        # Dodanie linii średniej i mediany
        ax.axvline(stats_data['Średnia'], color='green', linestyle=':', linewidth=1, label='Średnia')
        ax.axvline(stats_data['Mediana'], color='purple', linestyle=':', linewidth=1, label='Mediana')

        # Dodanie statystyk do wykresu
        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Koszty: {stats_data['Średnia kosztów']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}\n")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8))

        # Ustawienie etykiet osi
        ax.set_xlabel('Końcowy P/L' if idx >= 4 else '')
        ax.set_ylabel('Procent ścieżek [%]' if idx % 2 == 0 else '')

        # Dodanie legendy tylko do pierwszego wykresu
        if idx == 0:
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig('gamma_2_rehedge_margin_analysis_call_2.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[
        ['Rehedge Margin', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df


def gamma3_rehedge_margin_analysis(paths_gbm, params, rehedge_margins=None, K=1700, option_type='call',
                            save=False):
    """
    Analiza wpływu parametru rehedge_margin na rozkład P&L w strategii delta-gamma hedgingu.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    rehedge_margins: list
        Lista wartości parametru rehedge_margin do analizy
    K: float
        Cena wykonania dla opcji waniliowej
    option_type: str
        Typ opcji ('call' lub 'put')
    binary_option_type: str
        Typ opcji binarnej ('binary_call' lub 'binary_put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości rehedge_margin
    """

    if rehedge_margins is None or len(rehedge_margins) != 6:
        # Domyślne wartości rehedge_margin
        rehedge_margins = [0.1, 1, 3, 5, 10, 15]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle('Analiza wpływu parametru rehedge_margin na rozkład P&L (Delta-Gamma Hedging z dynamicznymi dwiema opcjami binarnymi)',
                 y=1.02, fontsize=14)

    # Spłaszczenie tablicy axes dla łatwiejszego iterowania
    axes_flat = axes.flatten()

    for idx, margin in enumerate(rehedge_margins):
        # Obliczenie P&L przy użyciu strategii delta-gamma
        pnl, portfolio_composition = wallet_delta_gamma_3(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            option_type=option_type,
            n_hedge=52,
            rehedge_margin=margin,
            delta_margin=0.1,  # Stała wartość
            denom_margin=1e-3  # Stała wartość
        )

        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        mask_nan = np.isnan(final_pnl)
        stats_data = {
            'Rehedge Margin': margin,
            'Średnia': np.nanmean(final_pnl),
            'Średnia kosztów': np.nanmean(portfolio_composition['cost']),
            'Mediana': np.nanmedian(final_pnl),
            'Std': np.nanstd(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl[~mask_nan]),
            'Kurtosis': scipy_stats.kurtosis(final_pnl[~mask_nan]),
            'Min': np.min(final_pnl[~mask_nan]),
            'Max': np.max(final_pnl[~mask_nan]),
            'P(PL>0)': 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan]),
            'NA': np.isnan(final_pnl).sum()
        }
        results.append(stats_data)

        # Histogram P&L dla bieżącego rehedge_margin
        ax = axes_flat[idx]
        sns.histplot(final_pnl, kde=True, ax=ax, stat='percent', color='blue', bins=30)
        ax.set_title(f'rehedge_margin = {margin} - Histogram P&L', fontsize=12)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)

        # Dodanie linii średniej i mediany
        ax.axvline(stats_data['Średnia'], color='green', linestyle=':', linewidth=1, label='Średnia')
        ax.axvline(stats_data['Mediana'], color='purple', linestyle=':', linewidth=1, label='Mediana')

        # Dodanie statystyk do wykresu
        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Koszty: {stats_data['Średnia kosztów']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}\n")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8))

        # Ustawienie etykiet osi
        ax.set_xlabel('Końcowy P/L' if idx >= 4 else '')
        ax.set_ylabel('Procent ścieżek [%]' if idx % 2 == 0 else '')

        # Dodanie legendy tylko do pierwszego wykresu
        if idx == 0:
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig('gamma_3_rehedge_margin_analysis_call.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[
        ['Rehedge Margin', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df