import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


def option_gamma(S, K, t, r, sigma, option_type):
    """Oblicza gamma dla opcji waniliowej i binarnej"""
    if t <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type in ['call', 'put']:  # Gamma dla opcji waniliowej
        return norm.pdf(d1) / (S * sigma * np.sqrt(t))
    else:  # Gamma dla opcji binarnej
        gamma = (np.exp(-r * t) * d1 * norm.pdf(d2) / (sigma ** 2 * S ** 2 * t))
        return -gamma if option_type == 'binary_call' else gamma


def func_K(tau, K_vanilla, r, sigma):
    """Oblicza optymalne strike'y opcji binarnych"""
    max_val = K_vanilla * np.exp(- (r + 1.5 * sigma ** 2) * tau)
    A = sigma * np.sqrt(tau)
    K_call = max_val * np.exp(r * tau + A ** 2 + 0.5 * A * np.sqrt(A ** 2 + 4))
    K_put = max_val * np.exp(r * tau + A ** 2 - 0.5 * A * np.sqrt(A ** 2 + 4))
    return K_call, K_put, max_val



# Parametry
S0 = 2266.92
K_vanilla = 1700
T = 1.0  # Całkowity czas do wygaśnięcia
r = 0.0346
sigma = 0.2656

# Zakres cen aktywa
strikes = np.linspace(600, 3200, 100)

# Czas do wygaśnięcia - animacja codzienna
time_points = np.linspace(T, 1 / 252, 252)  # Codzienne kroki od T do 1 dnia

# Wstępne generowanie danych
data = {
    'time': [],
    'K_call': [],
    'K_put': [],
    'max_val': [],
    'vanilla_gamma': [],
    'binary_call_gamma': [],
    'binary_put_gamma': [],
    'scaled_call_gamma': [],
    'scaled_put_gamma': []
}

for t in time_points:
    # Oblicz strike'e binarne
    K_call, K_put, max_val = func_K(t, K_vanilla, r, sigma)

    # Oblicz gamma
    vanilla_gamma = np.array([option_gamma(s, K_vanilla, t, r, sigma, 'call') for s in strikes])
    binary_call_gamma = np.array([option_gamma(s, K_call, t, r, sigma, 'binary_call') for s in strikes])
    binary_put_gamma = np.array([option_gamma(s, K_put, t, r, sigma, 'binary_put') for s in strikes])

    ############## Oblicz współczynniki skalowania

    # Oblicz współczynniki skalowania
    idx_max_vanilla = np.argmax(np.abs(vanilla_gamma))
    idx_max_call = np.argmax(np.abs(binary_call_gamma))
    idx_max_put = np.argmax(binary_put_gamma)

    # Oblicz współczynniki skalowania na podstawie maksimów
    scale_call = vanilla_gamma[idx_max_vanilla] / binary_call_gamma[idx_max_call]
    scale_put = vanilla_gamma[idx_max_vanilla] / binary_put_gamma[idx_max_put]

    # Zapisz dane
    data['time'].append(t)
    data['K_call'].append(K_call)
    data['K_put'].append(K_put)
    data['max_val'].append(max_val)
    data['vanilla_gamma'].append(vanilla_gamma)
    data['binary_call_gamma'].append(binary_call_gamma)
    data['binary_put_gamma'].append(binary_put_gamma)
    data['scaled_call_gamma'].append([g * scale_call for g in binary_call_gamma])
    data['scaled_put_gamma'].append([g * scale_put for g in binary_put_gamma])

# Inicjalizacja wykresu
fig = go.Figure()

# Kolory
colors = {
    'vanilla': '#1f77b4',
    'binary_call': '#ff7f0e',
    'binary_put': '#2ca02c'
}

# Dodanie początkowych śladów
initial_data = data['vanilla_gamma'][0]
fig.add_trace(go.Scatter(
    x=strikes,
    y=initial_data,
    name='Vanilla Call Gamma',
    line=dict(color=colors['vanilla'], width=3)
))

fig.add_trace(go.Scatter(
    x=strikes,
    y=data['scaled_call_gamma'][0],
    name=f'Scaled Binary Call Gamma (K={data["K_call"][0]:.1f})',
    line=dict(color=colors['binary_call'], dash='dot')
))

fig.add_trace(go.Scatter(
    x=strikes,
    y=data['scaled_put_gamma'][0],
    name=f'Scaled Binary Put Gamma (K={data["K_put"][0]:.1f})',
    line=dict(color=colors['binary_put'], dash='dash')
))

# Przygotowanie ramek animacji
frames = []
for i, t in enumerate(time_points):
    frames.append(go.Frame(
        data=[
            go.Scatter(x=strikes, y=data['vanilla_gamma'][i]),
            go.Scatter(
                x=strikes,
                y=data['scaled_call_gamma'][i],
                name=f'Scaled Binary Call Gamma (K={data["K_call"][i]:.1f})'
            ),
            go.Scatter(
                x=strikes,
                y=data['scaled_put_gamma'][i],
                name=f'Scaled Binary Put Gamma (K={data["K_put"][i]:.1f})'
            )
        ],
        name=f"t={t:.3f}",
        traces=[0, 1, 2]
    ))

# Konfiguracja animacji
sliders = [{
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 16},
        'prefix': 'Time to expiry: ',
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
            'args': [[f.name], {
                'frame': {'duration': 0, 'redraw': True},
                'mode': 'immediate',
                'fromcurrent': True
            }],
            'label': f'{t:.3f}',
            'method': 'animate'
        } for t, f in zip(time_points, frames)]
}]

# Przyciski kontrolne i układ
fig.update_layout(
    title='Gamma opcji waniliowej i dynamicznie obliczanych binarnych',
    xaxis_title='Cena aktywa',
    yaxis_title='Wartość gamma',
    plot_bgcolor='white',
    height=800,
    width=1500,
    margin=dict(l=80, r=80, t=120, b=80),
    hovermode='x unified',
    sliders=sliders,
    updatemenus=[{
        'type': 'buttons',
        'buttons': [
            {
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 50, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 50}
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
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.frames = frames

# Dodatkowe adnotacje
fig.add_annotation(
    x=0.95, y=0.95,
    xref='paper', yref='paper',
    text="Dynamiczne strike'y binarne obliczane wg func_K()",
    showarrow=False,
    bgcolor='white'
)

# Zapis i wyświetlenie
fig.write_html('dynamic_gamma_comparison.html', auto_open=True)