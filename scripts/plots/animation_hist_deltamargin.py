from delta_hedge.paths import generated_path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import skew, kurtosis
from delta_hedge.backtesting import *
from delta_hedge.simulation import *
from delta_hedge.hedging import wallet_delta_2

# Parametry początkowe
params = {
    's0': np.float64(2266.92),
    'volatility': np.float64(0.26538471196095936),
    'mu': np.float64(0.026072632409007104),
    'r': np.float64(0.03401487858727762)
}

# Lista delta_margin do analizy - TU MOŻESZ USTALIĆ WŁASNE WARTOŚCI
delta_margin_list = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Przykładowe wartości

# Stałe parametry symulacji
N = 1000  # Liczba ścieżek
T = 1  # Horyzont czasowy (w latach)
K = 1700  # Cena wykonania opcji
h = 1  # Krok czasowy
n_hedge = 52  # Stała liczba punktów hedgingu
rehedge_margin = 0.1  # Ustalona wartość rehedge_margin

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
def calculate_wallet_deltas(delta_margin_values):
    results = []
    for delta_margin in delta_margin_values:
        pnl, portfolio = wallet_delta_2(
            paths=paths_gbm,
            T_years=T,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            option_type='call',
            n_hedge=n_hedge,
            delta_margin=delta_margin,
            rehedge_margin=rehedge_margin
        )

        stats = {
            'delta_margin': delta_margin,
            'pnl': pnl,
            'mean': np.mean(pnl),
            'median': np.median(pnl),
            'std': np.std(pnl),
            'skewness': skew(pnl),
            'kurtosis': kurtosis(pnl),
            'min': np.min(pnl),
            'max': np.max(pnl),
            'cost': np.mean(portfolio['cost'])
        }
        results.append(stats)
    return results


# Obliczanie wyników
results = calculate_wallet_deltas(delta_margin_list)

# Przygotowanie danych
all_pnl = np.concatenate([res['pnl'] for res in results])
x_range_min = min(np.min(all_pnl), -np.max(np.abs(all_pnl)))
x_range_max = max(np.max(all_pnl), np.max(np.abs(all_pnl)))
n_bins = 50

# Tworzenie wykresu z czerwonymi słupkami
fig = go.Figure()

# Dodanie początkowego histogramu
initial_delta_margin = delta_margin_list[0]
initial_pnl = next(res['pnl'] for res in results if res['delta_margin'] == initial_delta_margin)

fig.add_trace(
    go.Histogram(
        x=initial_pnl,
        nbinsx=n_bins,
        name=f'Delta margin = {initial_delta_margin}',
        marker_color='#d62728',  # Czerwony
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
        f"Kurtoza: {res['kurtosis']:.2f}<br>"
        f"Koszt: {res['cost']:.2f}"
    )

    frames.append(
        go.Frame(
            data=[
                go.Histogram(
                    x=res['pnl'],
                    nbinsx=n_bins,
                    name=f'Delta margin = {res["delta_margin"]}',
                    marker_color='#d62728',
                    opacity=0.75
                )
            ],
            name=str(res['delta_margin']),
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
        'prefix': 'Delta margin = ',
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
                [str(res['delta_margin'])],
                {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'fromcurrent': True
                }
            ],
            'label': f'{res["delta_margin"]}',
            'method': 'animate'
        }
        for res in results
    ]
}]

# Przyciski kontrolne i układ wykresu
initial_stats = next(res for res in results if res['delta_margin'] == initial_delta_margin)
initial_stats_text = (
    f"<b>Statystyki:</b><br>"
    f"Średnia: {initial_stats['mean']:.2f}<br>"
    f"Mediana: {initial_stats['median']:.2f}<br>"
    f"Std: {initial_stats['std']:.2f}<br>"
    f"Skośność: {initial_stats['skewness']:.2f}<br>"
    f"Kurtoza: {initial_stats['kurtosis']:.2f}"
)

fig.update_layout(
    title=f'Histogram P&L dla różnych delta_margin (rehedge_margin={rehedge_margin})',
    xaxis_title='PNL',
    yaxis_title='Liczba wystąpień',
    bargap=0.1,
    sliders=sliders,
    updatemenus=[{
        'type': 'buttons',
        'buttons': [
            {
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 300}
                }]
            },
            {
                'label': '❚❚ Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 10},
        'x': 0.1,
        'y': 1.1,
        'xanchor': 'right',
        'yanchor': 'top'
    }],
    plot_bgcolor='white',
    height=800,
    width=1500,
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
pio.write_html(fig, file=generated_path('pnl_histogram_delta_margin.html'), auto_open=True)