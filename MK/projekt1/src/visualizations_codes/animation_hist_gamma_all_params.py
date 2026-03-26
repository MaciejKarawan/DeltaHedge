import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import skew, kurtosis
from src.backtesting import *
from src.simulation import *
from src.hedging import wallet_delta_gamma
import multiprocessing as mp
from functools import partial
import time
import os


def process_combination(args, paths, T, sigma, r, K, n_hedge):
    """Funkcja przetwarzająca pojedynczą kombinację parametrów"""
    binary_type, K_binary, rehedge_margin, delta_margin = args

    # Ustalona wartość dla binary_gamma_margin
    binary_gamma_margin = 1e-3

    # Obliczamy PnL dla danej kombinacji parametrów
    pnl, portfolio = wallet_delta_gamma(
        paths=paths,
        T_years=T,
        sigma=sigma,
        r=r,
        K=K,
        K_binary=K_binary,
        option_type='call',
        n_hedge=n_hedge,
        binary_option_type=binary_type,
        delta_margin=delta_margin,
        rehedge_margin=rehedge_margin,
        binary_gamma_margin=binary_gamma_margin
    )

    # Tworzymy statystyki
    stats = {
        'delta_margin': delta_margin,
        'rehedge_margin': rehedge_margin,
        'binary_gamma_margin': binary_gamma_margin,
        'K_binary': K_binary,
        'binary_option_type': binary_type,
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
    # Mierzenie czasu wykonania
    start_time = time.time()

    # Parametry początkowe
    params = {
        's0': np.float64(2266.92),
        'volatility': np.float64(0.26538471196095936),
        'mu': np.float64(0.026072632409007104),
        'r': np.float64(0.03401487858727762)
    }

    # Listy parametrów do analizy
    delta_margin_list = [0.0001, 0.001, 0.01, 0.1, 0.2]
    rehedge_margin_list = [0.1, 0.5, 0.7, 1, 1.5, 2, 3]
    K_binary_list = [1000, 1200, 1600, 1800, 2200, 2400]
    binary_option_types = ['binary_call', 'binary_put']

    # Ustalona wartość binary_gamma_margin
    binary_gamma_margin = 1e-3

    # Stałe parametry symulacji
    N = 1000  # Liczba ścieżek
    T = 1  # Horyzont czasowy (w latach)
    K = 1700  # Cena wykonania opcji
    h = 1  # Krok czasowy
    n_hedge = 52  # Stała liczba punktów hedgingu

    print("Generowanie ścieżek GBM...")
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

    # Obliczamy liczbę kombinacji - bez binary_gamma_margin
    total_combinations = len(binary_option_types) * len(K_binary_list) * len(rehedge_margin_list) * len(
        delta_margin_list)
    print(f"Łączna liczba kombinacji do obliczenia: {total_combinations}")

    # Przygotowanie listy wszystkich kombinacji parametrów - bez binary_gamma_margin
    all_combinations = [
        (binary_type, K_binary, rehedge_margin, delta_margin)
        for binary_type in binary_option_types
        for K_binary in K_binary_list
        for rehedge_margin in rehedge_margin_list
        for delta_margin in delta_margin_list
    ]

    # Funkcja częściowa z ustalonymi parametrami dla multiprocessing
    process_func = partial(
        process_combination,
        paths=paths_gbm,
        T=T,
        sigma=params['volatility'],
        r=params['r'],
        K=K,
        n_hedge=n_hedge
    )

    print("Rozpoczynam obliczenia wielowątkowe...")
    # Ustalenie liczby procesów - użycie 80% dostępnych rdzeni
    num_processes = max(1, int(mp.cpu_count() * 0.8))
    print(f"Liczba używanych procesów: {num_processes}")

    # Utworzenie pula procesów i uruchomienie obliczeń
    calculated_results = []
    completed = 0

    # Opcja 1: Użycie puli procesów do przetworzenia wszystkich kombinacji na raz
    with mp.Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_func, all_combinations)):
            args, stats = result
            calculated_results.append((args, stats))

            # Wyświetlenie postępu
            completed += 1
            if completed % 10 == 0 or completed == total_combinations:
                print(
                    f"Ukończono {completed}/{total_combinations} kombinacji ({completed / total_combinations * 100:.1f}%)")

    # Konwersja wyników na słownik 4D (bez binary_gamma_margin)
    print("Organizowanie wyników...")
    results_grid = {}
    all_results = []
    max_mean = -np.inf
    best_params = None

    for args, stats in calculated_results:
        binary_type, K_binary, rehedge_margin, delta_margin = args

        # Budowanie struktury 4D
        if binary_type not in results_grid:
            results_grid[binary_type] = {}
        if K_binary not in results_grid[binary_type]:
            results_grid[binary_type][K_binary] = {}
        if rehedge_margin not in results_grid[binary_type][K_binary]:
            results_grid[binary_type][K_binary][rehedge_margin] = {}

        # Zapisanie wyników
        results_grid[binary_type][K_binary][rehedge_margin][delta_margin] = stats
        all_results.append(stats)

        # Sprawdzenie czy to najlepszy wynik
        if stats['mean'] > max_mean:
            max_mean = stats['mean']
            best_params = (delta_margin, rehedge_margin, K_binary, binary_type)

    # Wyświetlenie informacji o najlepszej kombinacji
    print(
        f"Najlepsza kombinacja: delta={best_params[0]}, rehedge={best_params[1]}, K_binary={best_params[2]}, type={best_params[3]}")
    print(f"Średnia dla najlepszej kombinacji: {max_mean:.4f}")

    # Przygotowanie danych do wykresu
    all_pnl = np.concatenate([res['pnl'] for res in all_results])
    x_range_min = min(np.min(all_pnl), -np.max(np.abs(all_pnl)))
    x_range_max = max(np.max(all_pnl), np.max(np.abs(all_pnl)))
    n_bins = 50

    # Inicjalizacja parametrów dla pierwszego wykresu
    initial_binary_type = binary_option_types[0]
    initial_K_binary = K_binary_list[0]
    initial_rehedge_margin = rehedge_margin_list[0]
    initial_delta_margin = delta_margin_list[0]

    # Tworzenie wykresu
    print("Tworzenie wykresu interaktywnego...")
    fig = go.Figure()

    # Dodanie początkowego histogramu
    initial_pnl = results_grid[initial_binary_type][initial_K_binary][initial_rehedge_margin][initial_delta_margin][
        'pnl']

    fig.add_trace(
        go.Histogram(
            x=initial_pnl,
            nbinsx=n_bins,
            name=f'Initial Parameters',
            marker_color='#d62728',  # Czerwony
            opacity=0.75
        )
    )

    # Ustalenie zakresu osi
    fig.update_xaxes(range=[x_range_min, x_range_max])

    # Przygotowanie ramek animacji
    print("Generowanie ramek animacji...")
    frames = []
    for binary_type in binary_option_types:
        for K_binary in K_binary_list:
            for rehedge_margin in rehedge_margin_list:
                for delta_margin in delta_margin_list:
                    res = results_grid[binary_type][K_binary][rehedge_margin][delta_margin]
                    stats_text = (
                        f"<b>Statystyki:</b><br>"
                        f"Średnia: {res['mean']:.2f}<br>"
                        f"Mediana: {res['median']:.2f}<br>"
                        f"Std: {res['std']:.2f}<br>"
                        f"Koszt: {res['cost']:.2f}<br><br>"
                        f"<b>Najlepsza kombinacja:</b><br>"
                        f"Delta margin: {best_params[0]}<br>"
                        f"Rehedge margin: {best_params[1]}<br>"
                        f"K binary: {best_params[2]}<br>"
                        f"Binary option type: {best_params[3]}<br>"
                        f"Średnia: {max_mean:.2f}"
                    )

                    # Unikatowa nazwa ramki dla tej kombinacji parametrów - bez binary_gamma_margin
                    frame_name = f"{binary_type}-{K_binary}-{rehedge_margin}-{delta_margin}"

                    frames.append(
                        go.Frame(
                            data=[
                                go.Histogram(
                                    x=res['pnl'],
                                    nbinsx=n_bins,
                                    name=f'Parameters: {frame_name}',
                                    marker_color='#d62728',
                                    opacity=0.75
                                )
                            ],
                            name=frame_name,
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

    # Konfiguracja suwaków i przełącznika binary_option_type - bez binary_gamma_margin_slider
    # Przełącznik binary_option_type
    binary_type_dropdown = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 10},
            'prefix': 'Binary Option Type: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 100},
        'pad': {'b': 10, 't': 280},  # Dostosowanie pozycji po usunięciu jednego suwaka
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # Suwak K_binary
    K_binary_slider = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 10},
            'prefix': 'K binary = ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 100},
        'pad': {'b': 10, 't': 200},  # Dostosowanie pozycji po usunięciu jednego suwaka
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # Suwak rehedge_margin
    rehedge_margin_slider = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 10},
            'prefix': 'Rehedge margin = ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 100},
        'pad': {'b': 10, 't': 120},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # Suwak delta_margin
    delta_margin_slider = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 10},
            'prefix': 'Delta margin = ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 100},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # Dodanie kroków do suwaków i przełącznika
    for i, binary_type in enumerate(binary_option_types):
        binary_type_dropdown['steps'].append({
            'args': [
                [f"{binary_type}-{K_binary_list[0]}-{rehedge_margin_list[0]}-{delta_margin_list[0]}"],
                {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'fromcurrent': True
                }
            ],
            'label': f'{binary_type}',
            'method': 'animate'
        })

    for i, K_binary in enumerate(K_binary_list):
        K_binary_slider['steps'].append({
            'args': [
                [f"{binary_option_types[0]}-{K_binary}-{rehedge_margin_list[0]}-{delta_margin_list[0]}"],
                {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'fromcurrent': True
                }
            ],
            'label': f'{K_binary}',
            'method': 'animate'
        })

    for i, rehedge_margin in enumerate(rehedge_margin_list):
        rehedge_margin_slider['steps'].append({
            'args': [
                [f"{binary_option_types[0]}-{K_binary_list[0]}-{rehedge_margin}-{delta_margin_list[0]}"],
                {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'fromcurrent': True
                }
            ],
            'label': f'{rehedge_margin}',
            'method': 'animate'
        })

    for i, delta_margin in enumerate(delta_margin_list):
        delta_margin_slider['steps'].append({
            'args': [
                [f"{binary_option_types[0]}-{K_binary_list[0]}-{rehedge_margin_list[0]}-{delta_margin}"],
                {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'fromcurrent': True
                }
            ],
            'label': f'{delta_margin}',
            'method': 'animate'
        })

    # Funkcja do tworzenia przycisku dla wyboru konkretnej kombinacji parametrów
    def create_frame_selection_button(binary_type, K_binary, rehedge_margin, delta_margin):
        return {
            'args': [
                [f"{binary_type}-{K_binary}-{rehedge_margin}-{delta_margin}"],
                {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                }
            ],
            'label': f'Best combo',
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

    # Dodanie przycisku dla najlepszej kombinacji
    update_buttons.append(
        create_frame_selection_button(best_params[3], best_params[2], best_params[1], best_params[0]))

    # Statystyki początkowe
    initial_stats = results_grid[initial_binary_type][initial_K_binary][initial_rehedge_margin][initial_delta_margin]
    initial_stats_text = (
        f"<b>Statystyki:</b><br>"
        f"Średnia: {initial_stats['mean']:.2f}<br>"
        f"Mediana: {initial_stats['median']:.2f}<br>"
        f"Std: {initial_stats['std']:.2f}<br>"
        f"Koszt: {initial_stats['cost']:.2f}<br><br>"
        f"<b>Najlepsza kombinacja:</b><br>"
        f"Delta margin: {best_params[0]}<br>"
        f"Rehedge margin: {best_params[1]}<br>"
        f"K binary: {best_params[2]}<br>"
        f"Binary option type: {best_params[3]}<br>"
        f"Średnia: {max_mean:.2f}"
    )

    # Finalizacja układu wykresu
    fig.update_layout(
        title='Histogram P&L dla różnych kombinacji parametrów (binary_gamma_margin=1e-3)',
        xaxis_title='PNL',
        yaxis_title='Liczba wystąpień',
        bargap=0.1,
        sliders=[binary_type_dropdown, K_binary_slider, rehedge_margin_slider, delta_margin_slider],
        # Bez binary_gamma_margin_slider
        updatemenus=[{
            'type': 'buttons',
            'buttons': update_buttons,
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'x': 0.1,
            'y': 1.2,
            'xanchor': 'right',
            'yanchor': 'top'
        }],
        plot_bgcolor='white',
        height=800,  # Zmniejszona wysokość po usunięciu jednego suwaka
        width=1500,
        margin=dict(l=80, r=80, t=160, b=120),  # Dostosowany górny margines po usunięciu jednego suwaka
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

    # Skrypt JS do synchronizacji suwaków
    slider_js_script = """
    <script>
    // Kod do synchronizacji suwaków
    var graphDiv = document.getElementById('graph');
    var sliders = graphDiv.layout.sliders;

    // Funkcja pomocnicza do pobierania aktualnych wartości ze wszystkich suwaków
    function getCurrentSliderValues() {
        return sliders.map(function(slider, idx) {
            return slider.steps[slider.active].label;
        });
    }

    // Dla każdego suwaka
    sliders.forEach(function(slider, sliderIndex) {
        slider.steps.forEach(function(step, stepIndex) {
            // Tworzenie nowej funkcji dla każdego kroku suwaka
            step.args = [
                function() {
                    // Pobierz aktualne wartości wszystkich suwaków
                    var currentValues = getCurrentSliderValues();

                    // Aktualizuj wartość dla aktualnego suwaka
                    currentValues[sliderIndex] = slider.steps[stepIndex].label;

                    // Tworzenie nowej nazwy ramki
                    var newFrameName = currentValues.join('-');

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
    html_content = pio.to_html(fig, include_plotlyjs=True, full_html=True)
    html_content = html_content.replace('</body>', slider_js_script + '</body>')

    # Zapisujemy plik
    try:
        with open('pnl_histogram_gamma_fixed.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        print("Plik zapisany z powodzeniem w kodowaniu UTF-8")
    except Exception as e:
        print(f"Błąd podczas zapisu pliku: {e}")

        # Wersja bezpieczniejsza - standardowa metoda Plotly
        try:
            pio.write_html(fig, file='pnl_histogram_gamma_fixed_safe.html', auto_open=True)
            print("Plik zapisany z powodzeniem przy użyciu standardowej metody Plotly")
        except Exception as e2:
            print(f"Błąd podczas standardowego zapisu Plotly: {e2}")

    # Czas wykonania
    elapsed_time = time.time() - start_time
    print(f"Całkowity czas wykonania: {elapsed_time:.2f} sekund")


if __name__ == "__main__":
    # Ustawienie metody startowej dla multiprocessing
    mp.set_start_method('spawn', force=True)
    main()