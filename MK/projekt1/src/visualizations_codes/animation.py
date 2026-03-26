import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
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
import plotly.graph_objects as go
import plotly.io as pio

params = {'s0': np.float64(2266.92), 'volatility': np.float64(0.26538471196095936), 'mu': np.float64(0.026072632409007104), 'r': np.float64(0.03401487858727762)}

N = 1000
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

# 1. Najpierw generujemy i zapisujemy dane dla różnych wartości sigma
K_values = [1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600]
sigma_values = np.arange(0.75, 1.26, 0.01)*params['volatility']



# Inicjalizacja słownika do przechowywania wyników
all_results = {}
# Generowanie danych dla każdej kombinacji sigma i K
for sigma in sigma_values:
    results = []

    for K in K_values:
        # Twoja funkcja wallet_delta
        pnl, _ = wallet_delta(paths=paths_gbm, T_years=1, sigma=sigma, r=params['r'], K=K,
                              option_type='call', n_hedge=n_hedge)

        pnl_final = pnl

        # Obliczanie statystyk
        stats = {
            'K': K,
            'sigma': sigma,
            'q25': np.percentile(pnl_final, 25),
            'q50': np.percentile(pnl_final, 50),  # mediana
            'q75': np.percentile(pnl_final, 75),
            'mean': np.mean(pnl_final)
        }

        results.append(stats)

    # Zapisujemy wyniki dla danego sigma
    all_results[f'sigma_{sigma}'] = results

# Zapisz dane do formatu CSV lub pickle dla późniejszego użycia
results_df = pd.DataFrame([item for sublist in all_results.values() for item in sublist])
results_df.to_csv('pnl_results.csv', index=False)


# Wczytaj dane
results_df = pd.read_csv('pnl_results.csv')  # Zakładając, że mamy już zapisane dane


# Uzyskaj unikalne wartości sigma
sigma_values = sorted(results_df['sigma'].unique())

# Kolory dla poszczególnych serii danych
colors = {
    'q25': 'blue',
    'q50': 'green',
    'q75': 'red',
    'mean': 'black'
}

# Style linii
line_styles = {
    'q25': None,
    'q50': None,
    'q75': None,
    'mean': 'dash'
}

# Nazwy serii
series_names = {
    'q25': 'Kwantyl 25%',
    'q50': 'Mediana',
    'q75': 'Kwantyl 75%',
    'mean': 'Średnia'
}


# Tworzenie pustego wykresu
fig = go.Figure()

# Dodajemy początkowe ślady (dla pierwszej wartości sigma)
sigma_init = sigma_values[0]
sigma_data = results_df[results_df['sigma'] == sigma_init].sort_values('K')

for stat in ['q25', 'q50', 'q75', 'mean']:
    fig.add_trace(go.Scatter(
        x=sigma_data['K'],
        y=sigma_data[stat],
        mode='lines',  # Usuwamy markery dla lepszej wydajności
        name=series_names[stat],
        line=dict(
            color=colors[stat],
            dash=line_styles[stat],
            width=3 if stat == 'q50' else 2
        )
    ))

# Przygotowanie ramek dla animacji - optymalizacja
frames = []
for sigma in sigma_values:
    sigma_data = results_df[results_df['sigma'] == sigma].sort_values('K')

    frame_data = []
    for stat in ['q25', 'q50', 'q75', 'mean']:
        frame_data.append(
            go.Scatter(
                x=sigma_data['K'],
                y=sigma_data[stat],
                mode='lines',  # Bez markerów dla lepszej wydajności
                line=dict(
                    color=colors[stat],
                    dash=line_styles[stat],
                    width=3 if stat == 'q50' else 2
                )
            )
        )

    frame = go.Frame(
        data=frame_data,
        name=str(sigma),
        traces=[0, 1, 2, 3]
    )
    frames.append(frame)

fig.frames = frames

# OPTYMALIZACJA: Lepsze ustawienia suwaka i przejść
sliders = [{
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 16},
        'prefix': 'σ = ',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 100},  # Krótszy czas przejścia
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': [
        {
            'args': [
                [str(sigma)],
                {
                    'frame': {'duration': 0, 'redraw': False},  # Kluczowa zmiana: redraw: False
                    'mode': 'immediate'
                }
            ],
            'label': f'{sigma:.2f}',
            'method': 'animate'
        }
        for sigma in sigma_values
    ]
}]

# Uproszczone przyciski kontrolne
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
                    'frame': {'duration': 500, 'redraw': False},  # Dłuższy czas animacji
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
    title='P&L portfela w zależności od K dla różnych zakładanych wartości σ',
    xaxis={'title': 'Wartość K'},
    yaxis={'title': 'PNL'},
    legend={'title': 'Statystyki'},
    sliders=sliders,
    updatemenus=updatemenus,
    plot_bgcolor='white',
    height=700,
    width=1000,
    margin=dict(l=80, r=80, t=100, b=80),
    # Optymalizacja układu
    hovermode='closest',  # Poprawia wydajność interakcji
    uirevision='constant'  # Zachowuje stan UI pomiędzy animacjami
)

# Zapisz zoptymalizowany wykres
pio.write_html(fig, file='pnl_animated_plot_optimized.html', auto_open=True)