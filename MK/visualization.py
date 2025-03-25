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


def setup_plotting_style():
    """Konfiguruje styl wykresów dla całego projektu."""
    plt.rc("figure", autolayout=True, figsize=(11.69, 8.27), titlesize=10, titleweight='bold')
    plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=16, titlepad=10)
    sns.set_theme(style='whitegrid', context='paper', palette='dark')


def plot_price_history(data, title=None, highlight_dates=None):
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
    fig, ax = plt.subplots(figsize=(12, 8))

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
    ax.set_xlabel("Data")
    ax.set_ylabel("WIG20")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"WIG20 {data.Date.iloc[0].strftime('%Y-%m-%d')} do {data.Date.iloc[-1].strftime('%Y-%m-%d')}")

    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_returns_distribution(returns, log_returns):
    """
    Porównuje rozkłady zwrotów z rozkładem normalnym.

    Parameters:
    -----------
    returns : Series
        Zwroty procentowe.
    log_returns : Series
        Zwroty logarytmiczne.
    """
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Porównanie zwrotów dziennych')

    # Scatterploty
    sns.scatterplot(x=range(len(returns)), y=returns, ax=ax[0, 0], color='red', label="Zwroty procentowe")
    sns.scatterplot(x=range(len(log_returns)), y=log_returns, ax=ax[1, 0], color='blue', label="Zwroty logarytmiczne")

    # Porównanie z rozkładem normalnym
    x = np.linspace(-4, 4, 100)
    ax[0, 1].plot(x, norm.pdf(x, 0, 1), 'r', label="N(0,1) PDF")
    ax[1, 1].plot(x, norm.pdf(x, 0, 1), 'r', label="N(0,1) PDF")

    # Znormalizowane histogramy
    sns.histplot((returns - returns.mean()) / returns.std(), kde=True, color='red', ax=ax[0, 1], label="Zwroty proc.")
    sns.histplot((log_returns - log_returns.mean()) / log_returns.std(), kde=True, color='blue', ax=ax[1, 1],
                 label="Zwroty log.")

    # Tytuły i legendy
    ax[0, 0].set_title("Zwroty procentowe")
    ax[1, 0].set_title("Zwroty logarytmiczne")
    ax[0, 1].set_title("Histogram zwrotów proc. vs N(0,1)")
    ax[1, 1].set_title("Histogram zwrotów log. vs N(0,1)")

    for i in range(2):
        for j in range(2):
            ax[i, j].legend()
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')


    plt.tight_layout()
    plt.show()


def plot_historical_volatility(data, n_years_list, window_size=252, highlight_2022=True, plot=True):
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
    if plot is True: plt.figure(figsize=(14, 8))

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
        plt.title(f'Historyczna zmienność roczna WIG20 dla różnych okresów')
        plt.xlabel('Data')
        plt.ylabel('Zmienność roczna (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Okres obliczania:')

        # Formatowanie osi X
        plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    return data  # Zwracamy dane z obliczonymi zmiennościami


def plot_historical_mean_returns(data, n_years_list, window_size=252, highlight_2022=True, plot=True):
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
    if plot is True: plt.figure(figsize=(14, 8))

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
        plt.title(f'Historyczne średnie roczne zwroty WIG20 dla różnych okresów')
        plt.xlabel('Data')
        plt.ylabel('Średni roczny zwrot (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Okres obliczania:')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Linia na poziomie 0%

        # Formatowanie osi X
        plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    return data  # Zwracamy dane z obliczonymi średnimi zwrotami


def plot_quantiles_vs_historical(paths_list, historical_data, labels, colors, plot_together=True,
                                 title="Kwantyle symulowanych ścieżek vs rzeczywista trajektoria"):
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
        plt.figure(figsize=(12, 8))

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
            plt.plot(time, quantiles_paths[0], color=color, ls=':', label=f'5% {label}')
            plt.plot(time, quantiles_paths[4], color=color, ls=':', label=f'95% {label}')

        plt.legend()
        plt.xlabel("Czas (dni)")
        plt.ylabel("WIG20")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    else:
        # Każda metoda na osobnym wykresie
        for i, (paths, label, color) in enumerate(zip(paths_list, labels, colors)):
            quantiles_paths = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
            time = np.arange(paths.shape[1])

            plt.figure(figsize=(12, 8))

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
            plt.show()


def plot_option_payoff_histogram(option_type, strike, paths, r=0.05, T=1, bins=50, figsize=(12, 8)):
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
        option_name = f"Opcja Call (Strike = {strike})"
    else:  # put
        payoffs = np.maximum(strike - final_prices, 0)
        option_name = f"Opcja Put (Strike = {strike})"

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
    axes[0].set_title(f'Rozkład końcowych cen aktywa bazowego')
    axes[0].set_xlabel('Cena aktywa')
    axes[0].set_ylabel('Częstość')
    axes[0].legend()

    # Histogram wypłat opcji
    sns.histplot(payoffs, bins=bins, kde=True, ax=axes[1])
    axes[1].axvline(mean_payoff, color='green', linestyle='--',
                    label=f'Średnia wypłata = {mean_payoff:.4f}')
    axes[1].axvline(mean_discounted, color='purple', linestyle=':',
                    label=f'Średnia zdyskontowana = {mean_discounted:.4f}')
    axes[1].set_title(f'Rozkład wypłat dla {option_name}')
    axes[1].set_xlabel('Wypłata (Payoff)')
    axes[1].set_ylabel('Częstość')
    axes[1].legend()

    # Dodanie tekstowego opisu statystyk
    plt.figtext(0.02, 0.02, f"Statystyki:\n"
                            f"- Średnia wypłata: {mean_payoff:.4f}\n"
                            f"- Średnia zdyskontowana: {mean_discounted:.4f}\n"
                            f"- Maksymalna wypłata: {max_payoff:.4f}\n"
                            f"- Opcje in-the-money: {positive_payoffs:.2f}%", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Zrób miejsce na tekst

    return fig


def plot_correlation_analysis(data, title=None):
    """
    Tworzy wizualizacje korelacji między zwrotami WIG20 i KGHM.

    Parameters:
    -----------
    data : DataFrame
        Dane z kolumnami 'WIG20_log_return' i 'KGHM_log_return'.
    title : str, optional
        Tytuł wykresu.
    """
    fig, ax = plt.subplots(figsize=(14, 12))

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


def plot_joint_distribution(paths, asset_names=None, density_type='scatter'):
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
    plt.figure(figsize=(12, 10))

    if density_type == 'scatter':
        plt.scatter(final_values_1, final_values_2, alpha=0.5)
        plt.title(f'Łączny rozkład końcowych wartości (korelacja: {correlation:.4f})')
    elif density_type == 'kde':
        sns.kdeplot(x=final_values_1, y=final_values_2, fill=True, levels=20, cmap="Blues")
        plt.title(f'Estymator gęstości jądrowej 2D (korelacja: {correlation:.4f})')
    elif density_type == 'hexbin':
        plt.hexbin(final_values_1, final_values_2, gridsize=30, cmap='Blues')
        plt.colorbar(label='Liczba punktów')
        plt.title(f'Heksagonalny binning (korelacja: {correlation:.4f})')

    plt.xlabel(asset_names[0])
    plt.ylabel(asset_names[1])
    plt.grid(True, alpha=0.3)

    # Dodaj marginal plots
    ax1 = plt.axes([0.15, 0.85, 0.7, 0.1])
    sns.histplot(final_values_1, kde=True, ax=ax1)
    ax1.set_xlabel('')
    ax1.set_title(f'Rozkład {asset_names[0]}')

    ax2 = plt.axes([0.85, 0.15, 0.1, 0.7])
    sns.histplot(y=final_values_2, kde=True, ax=ax2)
    ax2.set_ylabel('')
    ax2.set_title(f'Rozkład {asset_names[1]}')

    plt.tight_layout()
    plt.show()


def plot_pnl(pnl, title="Rozkład końcowego P/L"):
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
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(title, fontsize=16)

    # Obliczenie końcowego P/L
    final_pnl = pnl[:, -1] if pnl.ndim > 1 else pnl

    # Statystyki
    mu = np.mean(final_pnl)
    sigma = np.std(final_pnl)
    positive_paths_pct = 100 * np.sum(final_pnl > 0) / len(final_pnl)

    # Histogram z seaborn
    sns.histplot(final_pnl, kde=True, ax=ax)

    # Linie na histogramie
    ax.axvline(mu, color='r', linestyle='--', label=f"Średnia: ${mu:.2f}")
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, label="P&L = 0")

    # Dodanie statystyk do wykresu
    stats_text = (
        f"Średnia: ${mu:.2f}\n"
        f"Odchylenie std: ${sigma:.2f}\n"
        f"Ścieżki z zyskiem: {positive_paths_pct:.1f}%"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title("Rozkład końcowego P&L")
    ax.set_xlabel('Końcowy P/L')
    ax.set_ylabel('Gęstość')
    ax.legend()

    # Pozostałe statystyki
    stats = {
        'mean': mu,
        'std': sigma,
        'positive_paths_pct': positive_paths_pct,
        'skewness': scipy_stats.skew(final_pnl),
        'kurtosis': scipy_stats.kurtosis(final_pnl)
    }

    plt.tight_layout()
    return fig, stats

