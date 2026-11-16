from delta_hedge.paths import generated_path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from delta_hedge.backtesting import *
from delta_hedge.simulation import *

# Parametry początkowe
params = {
    's0': np.float64(2266.92),
    'volatility': np.float64(0.26538471196095936),  # Bazowa wartość volatility
    'mu': np.float64(0.026072632409007104),
    'r': np.float64(0.03401487858727762)
}

# Zakres volatility do modyfikacji - tutaj możesz ustawić własne wartości
volatility_range = np.arange(0.75, 1.26, 0.01)*params['volatility']

# Stałe parametry symulacji
N = 1000  # Liczba ścieżek
T = 1  # Horyzont czasowy (w latach)
K = 1800  # Cena wykonania opcji
h = 1  # ?
n_hedge = 252  # Liczba punktów hedgingu

# Generowanie ścieżek GBM (tylko raz, dla bazowej volatility)
paths_gbm, time_gbm = generate_gbm_paths(
    N=N,
    T=T,
    s0=params['s0'],
    mean=params['mu'],
    volatility=params['volatility'],
    h=h,
    plot=False
)


# Funkcja do obliczania wyników portfela - generuje dane tylko raz
def calculate_wallet_deltas(volatility_values):
    results = []

    for sigma in volatility_values:
        # Obliczanie PNL dla danej volatility
        pnl, _ = wallet_delta(
            paths=paths_gbm,
            T_years=T,
            sigma=sigma,
            r=params['r'],
            K=K,
            option_type='call',
            n_hedge=n_hedge
        )

        # Zbieranie statystyk
        stats = {
            'volatility': sigma,
            'pnl': pnl,
            'mean': np.mean(pnl),
            'std': np.std(pnl),
            'min': np.min(pnl),
            'max': np.max(pnl)
        }
        results.append(stats)

    return results


# Obliczanie wyników tylko raz (jak w kodzie bazowym)
results = calculate_wallet_deltas(volatility_range)

# Przygotowanie danych do histogramu
all_pnl = np.concatenate([res['pnl'] for res in results])
# Ustalamy zakres osi X symetrycznie wokół zera lub zgodnie z danymi
x_range_min = min(np.min(all_pnl), -np.max(np.abs(all_pnl)))
x_range_max = max(np.max(all_pnl), np.max(np.abs(all_pnl)))
n_bins = 50  # Można dostosować

# Tworzenie wykresu
fig = go.Figure()

# Dodanie początkowego histogramu (dla pierwszej wartości volatility)
initial_volatility = volatility_range[0]
initial_pnl = next(res['pnl'] for res in results if res['volatility'] == initial_volatility)

fig.add_trace(
    go.Histogram(
        x=initial_pnl,
        nbinsx=n_bins,
        name=f'Volatility = {initial_volatility:.4f}',
        marker_color='blue',
        opacity=0.75
    )
)

# Ustalenie stałego zakresu osi X
fig.update_xaxes(range=[x_range_min, x_range_max])

# Przygotowanie ramek animacji
frames = []
for res in results:
    frames.append(
        go.Frame(
            data=[
                go.Histogram(
                    x=res['pnl'],
                    nbinsx=n_bins,
                    name=f'Volatility = {res["volatility"]:.4f}',
                    marker_color='blue',
                    opacity=0.75
                )
            ],
            name=str(res['volatility'])
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
        'prefix': 'Volatility = ',
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
                [str(res['volatility'])],
                {
                    'frame': {'duration': 0, 'redraw': True},  # Zmienione na True, aby naprawić problem z znikaniem
                    'mode': 'immediate',
                    'fromcurrent': True
                }
            ],
            'label': f'{res["volatility"]:.4f}',
            'method': 'animate'
        }
        for res in results
    ]
}]

# Przyciski kontrolne
updatemenus = [{
    'type': 'buttons',
    'showactive': False,
    'buttons': [
        {
            'label': '▶ Play',
            'method': 'animate',
            'args': [
                None,
                {
                    'frame': {'duration': 500, 'redraw': True},  # Zmienione na True
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

# Układ wykresu
fig.update_layout(
    title='Histogram P&L portfela dla różnych wartości volatility zakładanych w portfelu',
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
    uirevision='constant'  # Dodane dla stabilności UI
)

# Dodatkowe informacje w hover
fig.update_traces(
    hovertemplate='<b>Przedział</b>: %{x:.2f}<br><b>Liczba</b>: %{y}'
)


# Funkcja do dynamicznego aktualizowania zakresu
def update_xaxis_range():
    current_frame = fig.frames[fig.layout.sliders[0].active]
    current_data = current_frame.data[0].x
    return [min(np.min(current_data), x_range_min), max(np.max(current_data), x_range_max)]


from scipy.stats import skew, kurtosis


# Funkcja do obliczania statystyk
def calculate_stats(pnl):
    return {
        'mean': np.mean(pnl),
        'median': np.median(pnl),
        'std': np.std(pnl),
        'skewness': skew(pnl),
        'kurtosis': kurtosis(pnl)
    }


# Dodaj statystyki do każdej ramki
for i, res in enumerate(results):
    stats = calculate_stats(res['pnl'])
    stats_text = (
        f"<b>Statystyki:</b><br>"
        f"Średnia: {stats['mean']:.2f}<br>"
        f"Mediana: {stats['median']:.2f}<br>"
        f"Std: {stats['std']:.2f}<br>"
        f"Skośność: {stats['skewness']:.2f}<br>"
        f"Kurtoza: {stats['kurtosis']:.2f}"
    )

    fig.frames[i].layout.update(
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

# Dodaj początkową adnotację ze statystykami
initial_stats = calculate_stats(initial_pnl)
initial_stats_text = (
    f"<b>Statystyki:</b><br>"
    f"Średnia: {initial_stats['mean']:.2f}<br>"
    f"Mediana: {initial_stats['median']:.2f}<br>"
    f"Std: {initial_stats['std']:.2f}<br>"
    f"Skośność: {initial_stats['skewness']:.2f}<br>"
    f"Kurtoza: {initial_stats['kurtosis']:.2f}"
)

fig.update_layout(
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

# Zapisz wykres
pio.write_html(fig, file=generated_path('pnl_histogram_volatility.html'), auto_open=True)