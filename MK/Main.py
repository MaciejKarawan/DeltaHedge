import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import *
from src.simulation import *
from src.visualization import *
from src.pricing import *
from src.hedging import *
from src.backtesting import *



def run_part0():
    """Wykonuje analizy z części 0 (rozgrzewkowej)."""
    print("Uruchamiam część 0 - rozgrzewkowa")

    ################################## Wczytaj dane
    data_path = os.path.join('data', 'wig20_d_full.csv')
    kghm_path = os.path.join('data', 'kgh_d.csv')
    start_date = '01-01-2008'
    end_date = '01-01-2022'
    data, stats = load_data(data_path, start_date, end_date)



    ################################## Wykres danych historycznych
    #highlight_dates = {
    #    "2022-02-24": "Wybuch wojny na Ukrainie"
    #}
    #plot_price_history(data)




    ################################## Porównanie rozkładów zwrotów
    #compare_returns_distribution(stats['returns'], stats['log_returns'])



    ################################## Wykres volatility
    data1, stats1 = load_data(data_path, '01-01-2015', '01-01-2022')

    # Lista okresów w latach
    n_years_list = [1, 2, 3]
    data_with_volatility = plot_historical_volatility(data1, n_years_list, highlight_2022=False, plot=False)
    latest_data_vol = data_with_volatility.iloc[-1]

    # Print ostatnich wartości
    print("\nNajnowsze wartości zmienności:")
    for n in n_years_list:
        if f'volatility_{n}y' in data_with_volatility.columns:
            print(f"Zmienność w ujęciu rocznym zebrana z ostatnich {n}-lat: {latest_data_vol[f'volatility_{n}y']:.2f}%")

    ################################## Wykres mean
    data_with_means = plot_historical_mean_returns(data1, n_years_list, highlight_2022=False, plot=False)
    latest_data_mean = data_with_means.iloc[-1]

    # Print ostatnich wartości
    print("\nNajnowsze wartości średnich zwrotów:")
    for n in n_years_list:
        if f'mean_return_{n}y' in data_with_means.columns:
            print(f"Średni {n}-letni zwrot: {latest_data_mean[f'mean_return_{n}y']:.2f}%")
    print('\n')

    ################################## Symulacja ścieżek
    N = 10000
    T = 1 # wykres będzie miał t = 252 (wliczając s0)
    s0 = data1.iloc[-1, 4]  # data z 30-12-2021
    std = latest_data_vol[f'volatility_{2}y'] / 100 # Ustawiam zmienność roczną zebraną z ostatnich 2 lat
    mean = latest_data_mean[f'mean_return_{2}y'] / 100 # i średnią

    paths_gbm, time_gbm = generate_gbm_paths(
        N=N,
        T=T,
        s0=s0,
        mean=mean,
        volatility=std,
        h=1,
        plot=False
    )

    #### Test poprawności ścieżek
    #verify_gbm_paths(paths_gbm, s0, mean, std, T)

    paths_bootstrap, time_bootstrap = bootstrap_paths(
        N=N,
        T=T,
        s0=s0,
        historical_data=data1.loc[data1.Date >= '01-01-2020', :],
        plot=False
    )


    # Porównanie kwantyli z danymi historycznymi
    data2, _ = load_data(data_path, '30-12-2021', '01-01-2023')


    '''# Quantiles vs historical data
    plot_quantiles_vs_historical(
        paths_list=[paths_gbm, paths_bootstrap],
        historical_data=data2,
        labels=['GBM', 'Bootstrap'],
        colors=['blue', 'green'],
        plot_together=True,
        title="Kwantyle symulowanych ścieżek vs rzeczywista trajektoria"
    )'''



    # Rysuj histogramy dla opcji call i put

    strike = 1800
    r=0.05
    #fig_call = plot_option_payoff_histogram('call', strike, paths_gbm, r, T)
    #fig_put = plot_option_payoff_histogram('put', strike, paths_gbm, r, T)
    #plt.show()


    ########################### Analiza KGHM vs WIG20

    combined_data, stats = load_multiple_assets(data_path, kghm_path,
                                                         start_date='01-01-2020',
                                                         end_date='01-01-2022')
    #plot_correlation_analysis(combined_data)
    #plot_autocorrelation(combined_data, 60)
    #plt.show()

    ########################## Generowanie ścieżek za pomocą GBM

    '''
    s0_vec = [combined_data['WIG20'].iloc[-1], combined_data['KGHM'].iloc[-1]]
    mean_vec = [stats['wig20_mean'], stats['kghm_mean']]
    vol_vec = [stats['wig20_std'], stats['kghm_std']]

    # Macierz korelacji
    corr_matrix = np.array([
        [1.0, stats['correlation']],
        [stats['correlation'], 1.0]
    ])

    print("\nParametry symulacji:")
    print(f"Ceny początkowe: WIG20 = {s0_vec[0]:.2f}, KGHM = {s0_vec[1]:.2f} w momencie {combined_data.Date.iloc[0]}")
    print(f"Średnie zwroty (dzienne log): WIG20 = {stats['wig20_mean']:.6f}, KGHM = {stats['kghm_mean']:.6f}")
    print(f"Zmienność (dzienne log): WIG20 = {stats['wig20_std']:.6f}, KGHM = {stats['kghm_std']:.6f}")
    print(f"Korelacja: {stats['correlation']:.4f}")

    # Symulacja GBM
    print("\nGeneruję symulację GBM...")
    gbm_paths, time = generate_correlated_gbm_paths(
        N, T, s0_vec, mean_vec, vol_vec, corr_matrix
    )

    # Wizualizacja trajektorii
    plot_2d_paths(gbm_paths['asset_0'], gbm_paths['asset_1'], time, N, sample_paths=1)

    # Wizualizacja rozkładu końcowego
    print("\nTworzę wizualizację rozkładu końcowego GBM...")
    plot_joint_distribution(gbm_paths, ['asset_0', 'asset_1'], density_type='kde')
    '''

    ################## WALLET

    N = 10000
    T = 1
    r = np.log(1+0.05)# example
    h = 1
    n_hedge = 252 * 1

    paths_gbm, time_gbm = generate_gbm_paths(
        N=N,
        T=T,
        s0=s0,
        mean=mean,
        volatility=std,
        h = h,
        plot=False
    )

    #wallet_data, wallet_time = wallet(paths_gbm, 1, std, r, 1800, 'call', 252 * (1 / h))
    #fig, stats = plot_portfolio_pnl(np.cumsum(wallet_data, axis=1), wallet_time, num_paths_to_plot=10)
    pnl, time_series = wallet_delta(paths_gbm, 1, std, r, 1800, 'call', n_hedge)
    plot_pnl(pnl, title='Analiza Profit & Loss portfela')
    plt.show()

    print(f"Średni końcowy zysk/strata: ${np.mean(pnl):.2f}")
    print(f"Procent ścieżek z zyskiem: {np.mean(pnl > 0):.1f}%")

    '''
    ### Test portfela
    # Przykład testowania różnych częstotliwości rehedgingu
    n_hedge_values = [52, 252, 504, 1008]  # tygodniowo, dziennie, dwa razy dziennie, cztery razy dziennie
    results = {}

    for n_hedge in n_hedge_values:
        wallet_data, wallet_time = wallet_delta(paths_gbm, 1, std, r, strike, 'call', n_hedge)
        fig, stats = plot_pnl(wallet_data, f"Rozkład końcowego P/L (n_hedge={n_hedge})")

        plt.tight_layout()
        plt.show()
        #plt.savefig(f"pnl_distribution_nhedge_{n_hedge}.png")

        results[n_hedge] = stats
        print(f"\nStatystyki dla n_hedge={n_hedge}:")
        print(f"Średnia: {stats['mean']:.4f}")
        print(f"Odchylenie standardowe: {stats['std']:.4f}")
        print(f"Skośność: {stats['skewness']:.4f}")
        print(f"Kurtoza: {stats['kurtosis']:.4f}")
    '''

    print("Część 0 zakończona")


def run_partA():
    """Wykonuje analizy z części A."""
    #print("Uruchamiam część A")
    # Tu zaimplementujesz kod do części A
    pass


def run_partB():
    """Wykonuje analizy z części B."""
    #print("Uruchamiam część B")
    # Tu zaimplementujesz kod do części B
    pass


if __name__ == "__main__":
    # Konfiguracja stylu wykresów
    setup_plotting_style()

    # Uruchom analizy
    run_part0()
    run_partA()
    run_partB()