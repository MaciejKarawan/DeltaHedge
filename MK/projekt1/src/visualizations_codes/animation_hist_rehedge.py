import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import skew, kurtosis
from src.backtesting import *
from src.simulation import *

# Parametry początkowe
params = {
    's0': np.float64(2266.92),
    'volatility': np.float64(0.26538471196095936),
    'mu': np.float64(0.026072632409007104),
    'r': np.float64(0.03401487858727762)
}

# Lista n_hedge do analizy - TU MOŻESZ USTALIĆ WŁASNE WARTOŚCI
n_hedge_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 252]  # Przykładowe wartości

# Stałe parametry symulacji
N = 10000  # Liczba ścieżek
T = 1  # Horyzont czasowy (w latach)
K = 1800  # Cena wykonania opcji
h = 1  # Krok czasowy

# Generowanie ścieżek GBM (tylko raz)
paths_gbm, time_gbm = generate_gbm_paths(
    N=N,
    T=T,
    s0=params['s0'],
    mean=params['mu'],
    volatility=params['volatility'],
    h=h,
    plot=False
)


# Funkcja do obliczania wyników portfela
def calculate_wallet_deltas(n_hedge_values):
    results = []
    for n_hedge in n_hedge_values:
        pnl, _ = wallet_delta(
            paths=paths_gbm,
            T_years=T,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            option_type='call',
            n_hedge=n_hedge
        )

        stats = {
            'n_hedge': n_hedge,
            'pnl': pnl,
            'mean': np.mean(pnl),
            'median': np.median(pnl),
            'std': np.std(pnl),
            'skewness': skew(pnl),
            'kurtosis': kurtosis(pnl),
            'min': np.min(pnl),
            'max': np.max(pnl)
        }
        results.append(stats)
    return results


# Obliczanie wyników
results = calculate_wallet_deltas(n_hedge_list)

# Przygotowanie danych
all_pnl = np.concatenate([res['pnl'] for res in results])
x_range_min = min(np.min(all_pnl), -np.max(np.abs(all_pnl)))
x_range_max = max(np.max(all_pnl), np.max(np.abs(all_pnl)))
n_bins = 50

# Tworzenie wykresu z ciemnozielonymi słupkami
fig = go.Figure()

# Dodanie początkowego histogramu
initial_n_hedge = n_hedge_list[0]
initial_pnl = next(res['pnl'] for res in results if res['n_hedge'] == initial_n_hedge)

fig.add_trace(
    go.Histogram(
        x=initial_pnl,
        nbinsx=n_bins,
        name=f'Hedge points = {initial_n_hedge}',
        marker_color='#006d2c',  # Ciemna zieleń
        opacity=0.75
    )
)

# Ustalenie zakresu osi
fig.update_xaxes(range=[x_range_min, x_range_max])

# Przygotowanie ramek animacji
frames = []
for res in results:
    stats_text = (
        f"<b>Statystyki:</b><br>"
        f"Średnia: {res['mean']:.2f}<br>"
        f"Mediana: {res['median']:.2f}<br>"
        f"Std: {res['std']:.2f}<br>"
        f"Skośność: {res['skewness']:.2f}<br>"
        f"Kurtoza: {res['kurtosis']:.2f}"
    )

    frames.append(
        go.Frame(
            data=[
                go.Histogram(
                    x=res['pnl'],
                    nbinsx=n_bins,
                    name=f'Hedge points = {res["n_hedge"]}',
                    marker_color='#006d2c',
                    opacity=0.75
                )
            ],
            name=str(res['n_hedge']),
            layout=go.Layout(
                annotations=[
                    dict(
                        x=0.95,
                        y=0.95,
                        xref="paper",
                        yref="paper",
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

# Konfiguracja suwaka
sliders = [{
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 16},
        'prefix': 'Hedge points = ',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 100},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': [
        {
            'args': [
                [str(res['n_hedge'])],
                {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'fromcurrent': True
                }
            ],
            'label': f'{res["n_hedge"]}',
            'method': 'animate'
        }
        for res in results
    ]
}]

# Przyciski kontrolne
updatemenus = [{
    'type': 'buttons',
    'buttons': [
        {
            'label': '▶ Play',
            'method': 'animate',
            'args': [
                None,
                {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 300}
                }
            ]
        },
        {
            'label': '❚❚ Pause',
            'method': 'animate',
            'args': [
                [None],
                {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }
            ]
        }
    ],
    'direction': 'left',
    'pad': {'r': 10, 't': 10},
    'x': 0.1,
    'y': 1.1,
    'xanchor': 'right',
    'yanchor': 'top'
}]

# Układ wykresu z początkowymi statystykami
initial_stats = next(res for res in results if res['n_hedge'] == initial_n_hedge)
initial_stats_text = (
    f"<b>Statystyki:</b><br>"
    f"Średnia: {initial_stats['mean']:.2f}<br>"
    f"Mediana: {initial_stats['median']:.2f}<br>"
    f"Std: {initial_stats['std']:.2f}<br>"
    f"Skośność: {initial_stats['skewness']:.2f}<br>"
    f"Kurtoza: {initial_stats['kurtosis']:.2f}"
)

fig.update_layout(
    title='Histogram P&L portfela dla różnych częstotliwości hedgingu',
    xaxis_title='PNL',
    yaxis_title='Liczba wystąpień',
    bargap=0.1,
    sliders=sliders,
    updatemenus=updatemenus,
    plot_bgcolor='white',
    height=800,
    width=1400,
    margin=dict(l=80, r=80, t=120, b=80),
    hovermode='closest',
    uirevision='constant',
    annotations=[
        dict(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=initial_stats_text,
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

# Zapis wykresu
pio.write_html(fig, file='pnl_histogram_n_hedge.html', auto_open=True)