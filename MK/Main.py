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
from IPython.display import display



def run_part0():
    """Wykonuje analizy z części 0 (rozgrzewkowej)."""
    print("Uruchamiam część 0 - rozgrzewkowa")

    ################################## Wczytaj dane
    global data_path, kghm_path
    data_path = os.path.join('data', 'wig20_d_full.csv')
    kghm_path = os.path.join('data', 'kgh_d.csv')
    start_date = '01-01-2008'
    end_date = '01-01-2022'
    data, stats = load_data(data_path, start_date, end_date)

    ################################## Wykres danych historycznych
    #highlight_dates = {
    #    "2022-02-24": "Wybuch wojny na Ukrainie"
    #}
    #plot_price_history(data, title='Historyczny wykres zmiany ceny WIG20', save=False)

    ################################## Porównanie rozkładów zwrotów arytmetyczne vs logarytmiczne
    #compare_returns_distribution(stats['returns'], stats['log_returns'])


    ################################## Wykres volatility
    data1, stats1 = load_data(data_path, '01-01-2015', '01-01-2022')

    # Lista okresów w latach
    n_years_list = [1, 2, 3]
    data_with_volatility = plot_historical_volatility(data1, n_years_list, highlight_2022=False, plot=False, save=False)
    latest_data_vol = data_with_volatility.iloc[-1]

    # Print ostatnich wartości
    print("\nNajnowsze wartości zmienności:")
    for n in n_years_list:
        if f'volatility_{n}y' in data_with_volatility.columns:
            print(f"Zmienność w ujęciu rocznym zebrana z ostatnich {n}-lat: {latest_data_vol[f'volatility_{n}y']:.2f}%")

    ################################## Wykres mean
    data_with_means = plot_historical_mean_returns(data1, n_years_list, highlight_2022=False, plot=False, save=False)
    latest_data_mean = data_with_means.iloc[-1]

    # Print ostatnich wartości
    print("\nNajnowsze wartości średnich zwrotów:")
    for n in n_years_list:
        if f'mean_return_{n}y' in data_with_means.columns:
            print(f"Średni {n}-letni zwrot: {latest_data_mean[f'mean_return_{n}y']:.2f}%")
    print('\n')

    ################################## Symulacja ścieżek
    N = 10000
    T = 2 # wykres będzie miał t = 252 (wliczając s0)
    s0 = data1.iloc[-1, 4]  # data z 30-12-2021
    std = latest_data_vol[f'volatility_{2}y'] / 100 # Ustawiam zmienność roczną zebraną z ostatnich 2 lat
    mean = latest_data_mean[f'mean_return_{2}y'] / 100 # i średnią
    r = np.log(1 + 0.0346)

    global params
    params = {'s0': s0, 'volatility': std, 'mu': mean, 'r': r}
    #print(params)

    paths_gbm, time_gbm = generate_gbm_paths(
        N=N,
        T=T,
        s0=params['s0'],
        mean=params['mu'],
        volatility=params['volatility'],
        h=1,
        plot=False
    )

    #### Test poprawności ścieżek

    ## Porównanie kwantyli z danymi historycznymi
    global stats2
    data2, stats2 = load_data(data_path, '30-12-2021', '01-01-2023')
    #verify_gbm_paths(paths_gbm, s0, mean, std, T)
    simulated2022_returns = np.log(paths_gbm[0,:] / np.roll(paths_gbm[0, :], 1))[1:,]
    #compare_returns_distribution(stats2['log_returns'], simulated2022_returns, save=True)

    paths_bootstrap, time_bootstrap = bootstrap_paths(
        N=N,
        T=T,
        s0=params['s0'],
        historical_data=data1.loc[data1.Date >= '01-01-2020', :],
        plot=False
    )

    #verify_gbm_paths(paths_bootstrap, s0, mean, std, T)
    bootstrap_log_returns = np.log(paths_bootstrap[0,:] / np.roll(paths_bootstrap[0, :], 1))[1:,]
    #compare_returns_distribution(stats2['log_returns'], bootstrap_log_returns, save=True)

    # Quantiles vs historical data
    '''plot_quantiles_vs_historical(
        paths_list=[paths_gbm, paths_bootstrap],
        historical_data=data2,
        labels=['GBM', 'Bootstrap'],
        colors=['blue', 'green'],
        plot_together=True,
        title="Kwantyle symulowanych ścieżek vs rzeczywista trajektoria (2022 - 2023)",
        save=True
    )'''


    # Rysuj histogramy dla opcji call i put

    strike = 2600
    #fig_call = plot_option_payoff_histogram('call', strike, paths_gbm, 54.95, params['r'], T, save=True)
    #fig_put = plot_option_payoff_histogram('put', strike, paths_gbm, 363.35, params['r'], T, save=True)
    #plt.show()


    ########################### Analiza KGHM vs WIG20

    combined_data, stats = load_multiple_assets(data_path, kghm_path,
                                                         start_date='01-01-2020',
                                                         end_date='01-01-2022')
    #plot_correlation_analysis(combined_data, save=True)
    #plot_autocorrelation(combined_data, 60)
    #plt.show()

    ########################## Generowanie ścieżek za pomocą GBM


    '''s0_vec = [combined_data['WIG20'].iloc[-1], combined_data['KGHM'].iloc[-1]]
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
    #plot_2d_paths(gbm_paths['asset_0'], gbm_paths['asset_1'], time, N, sample_paths=1)

    # Wizualizacja rozkładu końcowego
    print("\nTworzę wizualizację rozkładu końcowego GBM...")
    plot_joint_distribution(gbm_paths, ['asset_0', 'asset_1'], density_type='kde', save=True)

'''
    print("Część 0 zakończona")


def run_partA():
    """Wykonuje analizy z części A."""
    ################## WALLET

    N = 10000
    T = 1
    K = 1800
    h = 1
    n_hedge = 252

    paths_gbm, time_gbm = generate_gbm_paths(
        N=N,
        T=T,
        s0=params['s0'],
        mean=params['mu'],
        volatility=params['volatility'],
        h=h,
        plot=False
    )

    pnl, portfolio_composition = wallet_delta(paths=paths_gbm,
                                              T_years=1,
                                              sigma=params['volatility'],
                                              r=params['r'],
                                              K=K,
                                              option_type='call',
                                              n_hedge=n_hedge)

    '''pnl, portfolio_composition = wallet_delta_2(paths=paths_gbm,
                                              T_years=1,
                                              sigma=params['volatility'],
                                              r=params['r'],
                                              K=K,
                                              option_type='call',
                                              n_hedge=n_hedge)'''

    ###### Overview portfela

    #fig, stats = plot_pnl(pnl, title='Histogram dla strategii delta hedgingu z kosztami', save=True, cost = portfolio_composition['cost'])
    #fig, stats = plot_qq_pnl(pnl, title="Q-Q plot dla strategii delta hedgingu z kosztami", save=True, cost = portfolio_composition['cost'])
    #plt.show()

    ###### rehedge_margin
    #fig, stats = delta_rehedge_margin_analysis(paths_gbm, params, rehedge_margins=None, K=1700, option_type='call', save=True)
    #plt.show()

    ###### TEST WALLET
    ### W zależności od liczby rehedge
    #fig, stats_df = perform_hedging_frequency_test(paths_gbm, params, K, save=True)
    #plt.show()
    #print("Statystyki:")
    #display(stats_df)

    ### W zależności od strike K
    fig = compare_pnl_by_strike(paths_gbm, params, strikes=[2300, 2800, 3300], option_type='call', save=True)
    plt.show()

    ### W zależności od volatility
    #print(f'Zaobserwowana zmienność roczna w 2022: {stats2['std']*np.sqrt(252)*100:.2f}% \n'
    #      f'Wyliczona zmienność roczna z okna dwuletniego 2020-2022: {params['volatility']*100:.2f}%')
    #print(stats2['std']*np.sqrt(252) / params['volatility'])
    #fig, stats = compare_pnl_by_volatility(paths_gbm, params, 1800, volatility=[params['volatility']*1, params['volatility']*1.1, params['volatility']*1.17], option_type='call', save=True)
    #plt.show()

    ### W zależności od średniej
    #fig, _ = compare_pnl_by_mu(10000, params, K, mu=[params['mu'] * 0.5, params['mu'], params['mu']*1.5], option_type='call', save=True)
    #plt.show()

    ### W zależności od stopy r
    #fig, stats = compare_pnl_by_r(paths_gbm, params, K, r=[params['r']*0.5, params['r'], params['r']*1.5], option_type='call', save=True)
    #plt.show()

    ### Premium
    # Zobaczmy jak wydarzenia w 2014 wpłynęły na zmienność na WIG20
    
    #data1, stats1 = load_data(data_path, '01-01-2010', '01-01-2017')
    #n_years_list = [1, 2, 3]
    #data_with_volatility = plot_historical_volatility(data1, n_years_list, highlight_2022=False, plot=True, save=True)
    
    # Zmienność licząc od połowy roku 2013 do połowy 2014 zmieniła się o 20-25%
    # Ciągnąć narrację o potencjalnej wojnie, policzymy premię od ryzyka przy tym założeniu

    #print(TVar_volrisk(paths_gbm, params, 1800))

    '''# Generowanie wykresu porównawczego
    fig = plot_extreme_paths_separately(
        portfolio_composition=portfolio_composition,
        pnl=pnl,
        n_extreme=1,  # Pokaż 5 największych zysków i 5 największych strat
        title="Analiza ekstremalnych scenariuszy hedgingu",
        save=True
    )
    plt.show()'''

    ### Wykres P/L w zależności od STRIKE
    #plot_pnl_by_K(paths_gbm = paths_gbm, params = params)

    plot_deltas(portfolio_composition['portfolio_delta'], 'Wykres delty portfolio dla Portfela Delta', 'Delta', True, 'delta')


def run_partB():
    """Wykonuje analizy z części B."""
    N = 10000
    T = 1
    K = 1700
    K_binary = 1200
    h = 1
    n_hedge = 52

    paths_gbm, time_gbm = generate_gbm_paths(
        N=N,
        T=T,
        s0=params['s0'],
        mean=params['mu'],
        volatility=params['volatility'],
        h=h,
        plot=False
    )

    ### DELTA-GAMMA 1 binary

    '''pnl, portfolio_composition = wallet_delta_gamma(paths_gbm, T, params['volatility'], params['r'],
                                                    K, K_binary, option_type='call', n_hedge=n_hedge,
                                                   binary_option_type='binary_put')'''


    #fig, stats = plot_pnl(pnl, title='Histogram dla strategii delta-gamma hedgingu z 1 opcją binarną put i kosztami', save=False, cost = portfolio_composition['cost'])
    #fig, stats = plot_qq_pnl(pnl, title='QQ-plot dla strategii delta-gamma hedgingu z 1 opcją binarną put i kosztami', save=True, cost = portfolio_composition['cost'])
    #plt.show()

    #fig, _ = gamma_bstrike(paths_gbm, params, binary_strikes=[1000, 1200, 1600, 1800, 2200, 2400],
    #             K=1700, option_type='call', binary_option_type='binary_call', save=False)
    #plt.show()

    '''fig, stats = gamma_rehedge_margin_analysis(paths_gbm, params, rehedge_margins=None, K=1700, option_type='call',
                            binary_option_type='binary_call',
                            save=True)
    plt.show()'''

    #fig, _ = compare_hedge_strategies(paths_gbm, params, K=1700, binary_strikes=[1200, 1600, 1700, 1800, 2300],
    #                         option_type='call', binary_option_type='binary_call', save=True)
    #plt.show()

    '''# Generowanie wykresu porównawczego
       fig = plot_extreme_paths_separately(
           portfolio_composition=portfolio_composition,
           pnl=pnl,
           n_extreme=1,  # Pokaż 5 największych zysków i 5 największych strat
           title="Analiza ekstremalnych scenariuszy hedgingu"
       )
       plt.show()'''

    #plot_deltas(portfolio_composition['delta'], 'Wykres delty portfolio dla Portfela Delta-Gamma', 'Delta', True, 'delta_gamma')
    #plot_deltas(portfolio_composition['gamma'], 'Wykres gammy portfolio dla Portfela Delta-Gamma', 'Gamma', True, 'delta_gamma')

    ### DELTA-GAMMA 2 binarki

    pnl, portfolio_composition = wallet_delta_gamma_2(paths_gbm,
                                                      T,
                                                      params['volatility'],
                                                      params['r'],
                                                      K,
                                                      K_binaries = [],
                                                      option_type='call', n_hedge=n_hedge)

    '''fig, stats = plot_pnl(pnl, title='Histogram dla strategii delta-gamma hedgingu z 2 opcjami binarnymi i kosztami', save=True,
                          cost=portfolio_composition['cost'])
    fig, stats = plot_qq_pnl(pnl, title='QQ-plot dla strategii delta-gamma hedgingu z 2 opcjami binarnymi i kosztami', save=True,
                             cost=portfolio_composition['cost'])
    plt.show()'''

    # fig, _ = gamma_bstrike(paths_gbm, params, binary_strikes=[1000, 1200, 1600, 1800, 2200, 2400],
    #             K=1700, option_type='call', binary_option_type='binary_call', save=False)
    # plt.show()

    #fig, stats = gamma2_rehedge_margin_analysis(paths_gbm, params, rehedge_margins=None, K=1700, option_type='call',
    #                        save=True)
    #plt.show()

    # fig, _ = compare_hedge_strategies(paths_gbm, params, K=1700, binary_strikes=[1200, 1600, 1700, 1800, 2300],
    #                         option_type='call', binary_option_type='binary_call', save=True)
    # plt.show()

    '''# Generowanie wykresu porównawczego
       fig = plot_extreme_paths_separately(
           portfolio_composition=portfolio_composition,
           pnl=pnl,
           n_extreme=1,  # Pokaż 5 największych zysków i 5 największych strat
           title="Analiza ekstremalnych scenariuszy hedgingu"
       )
       plt.show()'''

    #plot_deltas(portfolio_composition['delta'], 'Wykres delty portfolio dla Portfela Delta-Gamma-2', 'Delta', True, 'delta_gamma_2')
    #plot_deltas(portfolio_composition['gamma'], 'Wykres gammy portfolio dla Portfela Delta-Gamma-2', 'Gamma', True, 'delta_gamma_2')

    '''fig = plot_extreme_paths_separately(
        portfolio_composition=portfolio_composition,
        pnl=pnl,
        n_extreme=1,  # Pokaż 5 największych zysków i 5 największych strat
        title="Analiza ekstremalnych scenariuszy hedgingu"
    )'''
    #plt.show()

    pnl, portfolio_composition = wallet_delta_gamma_3(paths_gbm,
                                                      T,
                                                      params['volatility'],
                                                      params['r'],
                                                      K,
                                                      option_type='call', n_hedge=n_hedge)

    #plot_deltas(portfolio_composition['delta'], 'Wykres delty portfolio dla Portfela Delta-Gamma-3', 'Delta', True, 'delta_gamma_3')
    #plot_deltas(portfolio_composition['gamma'], 'Wykres gammy portfolio dla Portfela Delta-Gamma-3', 'Gamma', True, 'delta_gamma_3')

    #fig, stats = gamma3_rehedge_margin_analysis(paths_gbm, params, rehedge_margins=None, K=1700, option_type='call',
    #                       save=True)
    #plt.show()

    #fig, stats = plot_pnl(pnl, title='Histogram dla strategii delta-gamma hedgingu z dynamicznymi 2 opcjami binarnymi i kosztami', save=False,
    #                      cost=portfolio_composition['cost'])
    # fig, stats = plot_qq_pnl(pnl, title='QQ-plot dla strategii delta-gamma hedgingu z 2 opcjami binarnymi i kosztami', save=True,
    #                         cost=portfolio_composition['cost'])
    #plt.show()

    fig, stats = plot_pnl(pnl, title='Histogram dla strategii delta-gamma hedgingu dynamicznego z 2 opcjami binarnymi i kosztami', save=False,
                          cost=portfolio_composition['cost'])
    fig, stats = plot_qq_pnl(pnl, title='QQ-plot dla strategii delta-gamma hedgingu dynamicznego z 2 opcjami binarnymi i kosztami', save=False,
                             cost=portfolio_composition['cost'])
    plt.show()

if __name__ == "__main__":
    # Konfiguracja stylu wykresów
    setup_plotting_style()

    # Uruchom analizy
    run_part0()
    run_partA()
    #run_partB()