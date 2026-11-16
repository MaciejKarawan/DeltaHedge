from delta_hedge.paths import generated_path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import skew, kurtosis
from delta_hedge.backtesting import *
from delta_hedge.simulation import *
from delta_hedge.hedging import wallet_delta_gamma_2
import multiprocessing as mp
from functools import partial
import time


def process_combination(args, paths, T, sigma, r, K, n_hedge):
    """Funkcja przetwarzająca pojedynczą kombinację parametrów"""
    rehedge_margin, K_binary_call, K_binary_put, delta_margin = args

    # Obliczamy PnL dla danej kombinacji parametrów
    pnl, portfolio = wallet_delta_gamma_2(
        paths=paths,
        T_years=T,
        sigma=sigma,
        r=r,
        K=K,
        K_binaries=[K_binary_call, K_binary_put],
        option_type='call',
        n_hedge=n_hedge,
        delta_margin=delta_margin,
        rehedge_margin=rehedge_margin,
        denom_margin=1e-3
    )

    stats = {
        'delta_margin': delta_margin,
        'rehedge_margin': rehedge_margin,
        'K_binary_call': K_binary_call,
        'K_binary_put': K_binary_put,
        'pnl': pnl,
        'mean': np.mean(pnl),
        'median': np.median(pnl),
        'std': np.std(pnl),
        'skewness': skew(pnl),
        'kurtosis': kurtosis(pnl),
        'min': np.min(pnl),
        'max': np.max(pnl),
        'cost': np.mean(portfolio['cost']) if 'cost' in portfolio else np.nan
    }

    return args, stats


def main():
    start_time = time.time()

    # Parametry początkowe
    params = {
        's0': np.float64(2266.92),
        'volatility': np.float64(0.265),
        'mu': np.float64(0.026),
        'r': np.float64(0.034)
    }

    # Zakresy parametrów
    rehedge_margins = [0.1, 0.5, 1.0, 1.5, 2.0]
    K_binary_calls = [1000, 1200, 1500, 1900, 2300]
    K_binary_puts = [1000, 1200, 1500, 1900, 2300]
    delta_margins = [0.1]  # Ustalamy stałą wartość

    # Stałe parametry
    N = 1
    T = 1
    K = 1700
    n_hedge = 52

    # Generowanie ścieżek
    print("Generowanie ścieżek GBM...")
    paths_gbm, _ = generate_gbm_paths(
        N=N, T=T, s0=params['s0'],
        mean=params['mu'], volatility=params['volatility'], h=1 / T / n_hedge
    )

    # Przygotowanie kombinacji parametrów
    all_combinations = [
        (rehedge, K_call, K_put, delta)
        for rehedge in rehedge_margins
        for K_call in K_binary_calls
        for K_put in K_binary_puts
        for delta in delta_margins
    ]

    # Obliczenia równoległe
    print("Rozpoczynanie obliczeń równoległych...")
    num_processes = max(1, int(mp.cpu_count() * 0.8))
    process_func = partial(
        process_combination,
        paths=paths_gbm,
        T=T,
        sigma=params['volatility'],
        r=params['r'],
        K=K,
        n_hedge=n_hedge
    )

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_func, all_combinations)):
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"Ukończono {i + 1}/{len(all_combinations)} kombinacji")

    # Przygotowanie danych do wykresu
    results_dict = {}
    for args, stats in results:
        rehedge, K_call, K_put, delta = args
        key = f"{rehedge}-{K_call}-{K_put}-{delta}"
        results_dict[key] = stats

    # Tworzenie wykresu
    fig = go.Figure()

    # Dodanie początkowego histogramu
    initial_key = f"{rehedge_margins[0]}-{K_binary_calls[0]}-{K_binary_puts[0]}-{delta_margins[0]}"
    initial_pnl = results_dict[initial_key]['pnl']

    fig.add_trace(
        go.Histogram(
            x=initial_pnl,
            nbinsx=50,
            name='Initial',
            marker_color='#1f77b4',
            opacity=0.75
        )
    )

    # Przygotowanie ramek animacji
    frames = []
    for key in results_dict:
        rehedge, K_call, K_put, delta = key.split('-')
        stats = results_dict[key]

        stats_text = (
            f"<b>Parametry:</b><br>"
            f"Rehedge: {rehedge}<br>"
            f"K_call: {K_call}<br>"
            f"K_put: {K_put}<br>"
            f"Delta: {delta}<br><br>"
            f"<b>Statystyki:</b><br>"
            f"Średnia: {stats['mean']:.2f}<br>"
            f"Mediana: {stats['median']:.2f}<br>"
            f"Std: {stats['std']:.2f}<br>"
            f"Koszt: {stats['cost']:.2f}"
        )

        frames.append(
            go.Frame(
                data=[
                    go.Histogram(
                        x=stats['pnl'],
                        nbinsx=50,
                        name=key,
                        marker_color='#1f77b4',
                        opacity=0.75
                    )
                ],
                name=key,
                layout=go.Layout(
                    annotations=[
                        dict(
                            x=0.95, y=0.95,
                            xref="paper", yref="paper",
                            text=stats_text,
                            showarrow=False,
                            align="left",
                            bordercolor="#c7c7c7",
                            borderwidth=1,
                            borderpad=4,
                            bgcolor="#ffffff",
                            opacity=0.8
                        )
                    ]
                )
            )
        )

    fig.frames = frames

    # Konfiguracja suwaków
    sliders = []

    # Suwak rehedge_margin
    sliders.append({
        'active': 0,
        'currentvalue': {'prefix': 'Rehedge margin: '},
        'steps': [
            {
                'args': [
                    [f"{rehedge}-{K_binary_calls[0]}-{K_binary_puts[0]}-{delta_margins[0]}"],
                    {'frame': {'duration': 0, 'redraw': True}}
                ],
                'label': str(rehedge),
                'method': 'animate'
            }
            for rehedge in rehedge_margins
        ],
        'pad': {'t': 50}
    })

    # Suwak K_binary_call
    sliders.append({
        'active': 0,
        'currentvalue': {'prefix': 'K call: '},
        'steps': [
            {
                'args': [
                    [f"{rehedge_margins[0]}-{K_call}-{K_binary_puts[0]}-{delta_margins[0]}"],
                    {'frame': {'duration': 0, 'redraw': True}}
                ],
                'label': str(K_call),
                'method': 'animate'
            }
            for K_call in K_binary_calls
        ],
        'pad': {'t': 120}
    })

    # Suwak K_binary_put
    sliders.append({
        'active': 0,
        'currentvalue': {'prefix': 'K put: '},
        'steps': [
            {
                'args': [
                    [f"{rehedge_margins[0]}-{K_binary_calls[0]}-{K_put}-{delta_margins[0]}"],
                    {'frame': {'duration': 0, 'redraw': True}}
                ],
                'label': str(K_put),
                'method': 'animate'
            }
            for K_put in K_binary_puts
        ],
        'pad': {'t': 190}
    })

    # Przyciski kontrolne
    updatemenus = [{
        'type': 'buttons',
        'buttons': [
            {
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True
                }]
            },
            {
                'label': '❚❚ Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate'
                }]
            }
        ],
        'pad': {'r': 10, 't': 10},
        'x': 0.1,
        'y': 0,
        'xanchor': 'right',
        'yanchor': 'top'
    }]

    # Układ wykresu
    fig.update_layout(
        title='Analiza strategii Delta-Gamma z dwoma opcjami binarnymi',
        xaxis_title='P&L',
        yaxis_title='Liczba wystąpień',
        sliders=sliders,
        updatemenus=updatemenus,
        height=800,
        width=1200
    )

    # Zapis wykresu
    pio.write_html(fig, file=generated_path('delta_gamma_analysis.html'), auto_open=True)

    print(f"Czas wykonania: {time.time() - start_time:.2f} sekund")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()