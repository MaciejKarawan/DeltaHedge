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

# Lista delta_margin i rehedge_margin do analizy
delta_margin_list = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Przykładowe wartości
rehedge_margin_list = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Przykładowe wartości

# Stałe parametry symulacji
N = 10000  # Liczba ścieżek
T = 1  # Horyzont czasowy (w latach)
K = 1700  # Cena wykonania opcji
h = 1  # Krok czasowy
n_hedge = 52  # Stała liczba punktów hedgingu

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


# Funkcja do obliczania wyników portfela dla kombinacji parametrów
def calculate_wallet_deltas_grid(delta_margin_values, rehedge_margin_values):
    results_grid = {}
    all_results = []
    max_mean = -np.inf
    best_params = None

    for rehedge_margin in rehedge_margin_values:
        results_grid[rehedge_margin] = {}
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
                'rehedge_margin': rehedge_margin,
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

            results_grid[rehedge_margin][delta_margin] = stats
            all_results.append(stats)

            # Sprawdzenie czy mamy nowe najlepsze parametry
            if stats['mean'] > max_mean:
                max_mean = stats['mean']
                best_params = (delta_margin, rehedge_margin)

    return results_grid, all_results, best_params, max_mean


# Obliczanie wyników dla wszystkich kombinacji parametrów
results_grid, all_results, best_params, best_mean = calculate_wallet_deltas_grid(delta_margin_list, rehedge_margin_list)

# Przygotowanie danych do wykresu
all_pnl = np.concatenate([res['pnl'] for res in all_results])
x_range_min = min(np.min(all_pnl), -np.max(np.abs(all_pnl)))
x_range_max = max(np.max(all_pnl), np.max(np.abs(all_pnl)))
n_bins = 50

# Tworzenie wykresu
fig = go.Figure()

# Dodanie początkowego histogramu
initial_delta_margin = delta_margin_list[0]
initial_rehedge_margin = rehedge_margin_list[0]
initial_pnl = results_grid[initial_rehedge_margin][initial_delta_margin]['pnl']

fig.add_trace(
    go.Histogram(
        x=initial_pnl,
        nbinsx=n_bins,
        name=f'Delta margin = {initial_delta_margin}, Rehedge margin = {initial_rehedge_margin}',
        marker_color='#d62728',  # Czerwony
        opacity=0.75
    )
)

# Ustalenie zakresu osi
fig.update_xaxes(range=[x_range_min, x_range_max])

# Przygotowanie ramek animacji
frames = []
for rehedge_margin in rehedge_margin_list:
    for delta_margin in delta_margin_list:
        res = results_grid[rehedge_margin][delta_margin]
        stats_text = (
            f"<b>Statystyki:</b><br>"
            f"Średnia: {res['mean']:.2f}<br>"
            f"Mediana: {res['median']:.2f}<br>"
            f"Std: {res['std']:.2f}<br>"
            f"Skośność: {res['skewness']:.2f}<br>"
            f"Kurtoza: {res['kurtosis']:.2f}<br>"
            f"Koszt: {res['cost']:.2f}<br><br>"
            f"<b>Najlepsza kombinacja:</b><br>"
            f"Delta margin: {best_params[0]}<br>"
            f"Rehedge margin: {best_params[1]}<br>"
            f"Średnia: {best_mean:.2f}"
        )

        frames.append(
            go.Frame(
                data=[
                    go.Histogram(
                        x=res['pnl'],
                        nbinsx=n_bins,
                        name=f'Delta margin = {delta_margin}, Rehedge margin = {rehedge_margin}',
                        marker_color='#d62728',
                        opacity=0.75
                    )
                ],
                name=f"{delta_margin}-{rehedge_margin}",  # Nazwa ramki jako kombinacja obu parametrów
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

# Konfiguracja pierwszego suwaka dla delta_margin
delta_margin_slider = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 12},
        'prefix': 'Delta margin = ',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 100},
    'pad': {'b': 10, 't': 110},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# Konfiguracja drugiego suwaka dla rehedge_margin
rehedge_margin_slider = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 12},
        'prefix': 'Rehedge margin = ',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 100},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0.05,
    'steps': []
}

# Dodanie kroków do suwaków
for i, delta_margin in enumerate(delta_margin_list):
    delta_margin_slider['steps'].append({
        'args': [
            [f"{delta_margin}-{rehedge_margin_list[0]}"],
            {
                'frame': {'duration': 0, 'redraw': True},
                'mode': 'immediate',
                'fromcurrent': True
            }
        ],
        'label': f'{delta_margin}',
        'method': 'animate'
    })

for i, rehedge_margin in enumerate(rehedge_margin_list):
    rehedge_margin_slider['steps'].append({
        'args': [
            [f"{delta_margin_list[0]}-{rehedge_margin}"],
            {
                'frame': {'duration': 0, 'redraw': True},
                'mode': 'immediate',
                'fromcurrent': True
            }
        ],
        'label': f'{rehedge_margin}',
        'method': 'animate'
    })


# Funkcja do aktualizacji wykresu na podstawie obu parametrów
def create_frame_selection_button(delta_margin, rehedge_margin):
    return {
        'args': [
            [f"{delta_margin}-{rehedge_margin}"],
            {
                'frame': {'duration': 0, 'redraw': True},
                'mode': 'immediate',
            }
        ],
        'label': f'DM={delta_margin}, RM={rehedge_margin}',
        'method': 'animate'
    }


# Przyciski kontrolne
update_buttons = [
    {
        'label': '▶ Play All',
        'method': 'animate',
        'args': [None, {
            'frame': {'duration': 300, 'redraw': True},
            'fromcurrent': True,
            'transition': {'duration': 200}
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
]

# Dodanie przycisków dla najlepszej kombinacji
update_buttons.append(create_frame_selection_button(best_params[0], best_params[1]))

# Statystyki początkowe
initial_stats = results_grid[initial_rehedge_margin][initial_delta_margin]
initial_stats_text = (
    f"<b>Statystyki:</b><br>"
    f"Średnia: {initial_stats['mean']:.2f}<br>"
    f"Mediana: {initial_stats['median']:.2f}<br>"
    f"Std: {initial_stats['std']:.2f}<br>"
    f"Skośność: {initial_stats['skewness']:.2f}<br>"
    f"Kurtoza: {initial_stats['kurtosis']:.2f}<br>"
    f"Koszt: {initial_stats['cost']:.2f}<br><br>"
    f"<b>Najlepsza kombinacja:</b><br>"
    f"Delta margin: {best_params[0]}<br>"
    f"Rehedge margin: {best_params[1]}<br>"
    f"Średnia: {best_mean:.2f}"
)

# Finalizacja układu wykresu
fig.update_layout(
    title='Histogram P&L dla różnych kombinacji delta_margin i rehedge_margin',
    xaxis_title='PNL',
    yaxis_title='Liczba wystąpień',
    bargap=0.1,
    sliders=[delta_margin_slider, rehedge_margin_slider],
    updatemenus=[{
        'type': 'buttons',
        'buttons': update_buttons,
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
    margin=dict(l=80, r=80, t=120, b=120),  # Zwiększona dolna margines dla drugiego suwaka
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

# Zapis wykresu bez skryptów JS (czysta wersja)
pio.write_html(fig, file=generated_path('pnl_histogram_double_slider_basic.html'), auto_open=False)

# Zapis wykresu z dodatkowymi skryptami JS
# Użyj prostszej wersji skryptu, bez potencjalnie problematycznych znaków Unicode
slider_js_script = """
<script>
// Kod do synchronizacji obu suwaków
var graphDiv = document.getElementById('graph');
var sliders = graphDiv.layout.sliders;

// Dla każdego slidera 
sliders.forEach(function(slider, sliderIndex) {
    slider.steps.forEach(function(step, stepIndex) {
        var originalArgs = step.args[0];

        // Tworzenie nowej funkcji dla każdego kroku suwaka
        step.args = [
            function() {
                // Pobierz aktualną wartość drugiego suwaka
                var otherSliderIndex = sliderIndex === 0 ? 1 : 0;
                var otherSlider = sliders[otherSliderIndex];
                var otherValue = otherSlider.steps[otherSlider.active].label;

                // Tworzenie nowej nazwy ramki
                var newFrameName;
                if (sliderIndex === 0) {
                    // Jeśli aktualny suwak to delta_margin
                    var delta_margin = slider.steps[stepIndex].label;
                    newFrameName = delta_margin + '-' + otherValue;
                } else {
                    // Jeśli aktualny suwak to rehedge_margin
                    var rehedge_margin = slider.steps[stepIndex].label;
                    var otherSlider = sliders[0];
                    var delta_margin = otherSlider.steps[otherSlider.active].label;
                    newFrameName = delta_margin + '-' + rehedge_margin;
                }

                // Aktualizacja animacji
                Plotly.animate('graph', [newFrameName], {
                    frame: {duration: 0, redraw: true},
                    mode: 'immediate',
                    transition: {duration: 0}
                });

                return null;
            }
        ];
        step.execute = true;
    });
});
</script>
"""

# Generowanie pełnego HTML
html_content = pio.to_html(generated_path(fig), include_plotlyjs=True, full_html=True)
html_content = html_content.replace('</body>', slider_js_script + '</body>')

# Zapisujemy plik z jawnie określonym kodowaniem UTF-8
try:
    with open('pnl_histogram_double_slider.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Plik zapisany z powodzeniem w kodowaniu UTF-8")
except Exception as e:
    print(f"Błąd podczas zapisu pliku: {e}")

    # Alternatywna metoda zapisu - wersja tylko z ASCII
    try:
        # Usuwamy niestandardowe znaki
        html_content_ascii = html_content.encode('ascii', 'ignore').decode('ascii')
        with open('pnl_histogram_double_slider_ascii.html', 'w', encoding='ascii') as f:
            f.write(html_content_ascii)
        print("Plik zapisany z powodzeniem w kodowaniu ASCII (bez znaków diakrytycznych)")
    except Exception as e2:
        print(f"Błąd podczas zapisu pliku ASCII: {e2}")

        # Ostateczna próba - użyj standardowej metody Plotly
        try:
            pio.write_html(fig, file=generated_path('pnl_histogram_double_slider_safe.html'), auto_open=True)
            print("Plik zapisany z powodzeniem przy użyciu standardowej metody Plotly")
        except Exception as e3:
            print(f"Błąd podczas standardowego zapisu Plotly: {e3}")