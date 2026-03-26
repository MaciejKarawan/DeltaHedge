import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from src.backtesting import *
from src.simulation import *

# Parametry początkowe
params = {
    's0': np.float64(2266.92),
    'volatility': np.float64(0.26538471196095936),
    'mu': np.float64(0.026072632409007104),
    'r': np.float64(0.03401487858727762)  # Bazowa stopa procentowa
}

# Zakres stopy procentowej r do modyfikacji
r_range = np.arange(0.75, 1.5, 0.01)*params['r']

# Stałe parametry symulacji
N = 1000  # Liczba ścieżek
T = 1  # Horyzont czasowy
K = 1800  # Cena wykonania opcji
n_hedge = 252  # Liczba punktów hedgingu

# Generowanie ścieżek GBM (tylko raz)
print("Generowanie ścieżek GBM...")
paths_gbm, _ = generate_gbm_paths(
    N=N,
    T=T,
    s0=params['s0'],
    mean=params['mu'],
    volatility=params['volatility'],
    h=1,
    plot=False
)


# Funkcja do obliczania wyników portfela
def calculate_wallet_deltas(r_values):
    results = []
    print("Obliczanie wyników portfela dla różnych stóp r...")

    for r in r_values:
        pnl, _ = wallet_delta(
            paths=paths_gbm,
            T_years=T,
            sigma=params['volatility'],
            r=r,
            K=K,
            option_type='call',
            n_hedge=n_hedge
        )

        results.append({
            'r': r,
            'pnl': pnl,
            'mean': np.mean(pnl),
            'positive_ratio': np.mean(pnl > 0),
            'profit_prob': np.mean(pnl > np.percentile(pnl, 50))  # Prawdopodobieństwo zysku powyżej mediany
        })

    return results


# Obliczanie wyników
results = calculate_wallet_deltas(r_range)

# Przygotowanie danych do histogramu
all_pnl = np.concatenate([res['pnl'] for res in results])
x_range = [np.percentile(all_pnl, 1), np.percentile(all_pnl, 99)]  # Zakres od 1% do 99% percentyla
n_bins = 50

# Tworzenie wykresu
fig = go.Figure()

# Dodanie początkowego histogramu
initial_data = results[0]
fig.add_trace(
    go.Histogram(
        x=initial_data['pnl'],
        nbinsx=n_bins,
        name=f'r = {initial_data["r"]:.2%}',
        marker_color='#ff7f0e',
        opacity=0.75,
        hovertemplate='<b>Przedział</b>: %{x:.2f}<br><b>Częstość</b>: %{y}'
    )
)

# Ustawienia osi
fig.update_xaxes(range=x_range, title_text='PNL')
fig.update_yaxes(title_text='Liczba wystąpień')

# Przygotowanie ramek animacji
frames = []
for res in results:
    frames.append(
        go.Frame(
            data=[go.Histogram(
                x=res['pnl'],
                nbinsx=n_bins,
                name=f'r = {res["r"]:.2%}',
                marker_color='#ff7f0e'
            )],
            name=str(res['r']),
            layout=go.Layout(
                annotations=[
                    dict(
                        x=0.02,
                        y=0.95,
                        xref='paper',
                        yref='paper',
                        text=f"Stopa r: {res['r']:.2%}",
                        font=dict(size=14, color='black'),
                        bgcolor='white',
                        showarrow=False
                    ),
                    dict(
                        x=0.02,
                        y=0.85,
                        xref='paper',
                        yref='paper',
                        text=f"Dodatnie PNL: {res['positive_ratio']:.1%}",
                        font=dict(size=12),
                        showarrow=False
                    ),
                    dict(
                        x=0.02,
                        y=0.80,
                        xref='paper',
                        yref='paper',
                        text=f"Średni PNL: {res['mean']:.2f}",
                        font=dict(size=12),
                        showarrow=False
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
        'prefix': 'Stopa r = ',
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
                [str(res['r'])],
                {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'fromcurrent': True
                }
            ],
            'label': f'{res["r"]:.2%}',
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
}]

# Finalny układ wykresu
fig.update_layout(
    title='Rozkład PNL w zależności od stopy procentowej (r)',
    plot_bgcolor='white',
    height=750,
    width=1000,
    margin=dict(l=80, r=80, t=100, b=80),
    hovermode='closest',
    sliders=sliders,
    updatemenus=updatemenus,
    annotations=[
        dict(
            x=0.02,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f"Stopa r: {results[0]['r']:.2%}",
            font=dict(size=14, color='black'),
            bgcolor='white',
            showarrow=False
        ),
        dict(
            x=0.02,
            y=0.85,
            xref='paper',
            yref='paper',
            text=f"Dodatnie PNL: {results[0]['positive_ratio']:.1%}",
            font=dict(size=12),
            showarrow=False
        ),
        dict(
            x=0.02,
            y=0.80,
            xref='paper',
            yref='paper',
            text=f"Średni PNL: {results[0]['mean']:.2f}",
            font=dict(size=12),
            showarrow=False
        )
    ]
)

# Zapisz i pokaż wykres
print("Generowanie wykresu...")
pio.write_html(fig, file='pnl_vs_interest_rate.html', auto_open=True)
print("Gotowe!")