import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data
from simulation import generate_gbm_paths, bootstrap_paths
from visualization import *



plt.rc(
    "figure",
    autolayout=True,
    figsize=(11.69,8.27),
    titlesize=10,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
sns.set_theme(style='whitegrid',
              context='paper',
              palette='dark')


def find_series_lengths(lst):
    if not lst:
        return []

    result = []
    count = 1

    for i in range(1, len(lst)):
        if (lst[i] >= 0 and lst[i - 1] >= 0) or (lst[i] < 0 and lst[i - 1] < 0):
            count += 1
        else:
            result.append(count)
            count = 1  # Reset licznika dla nowej serii

    result.append(count)  # Dodajemy ostatnią serię
    return result

data_path = os.path.join('..','data', 'wig20_d_full.csv')
data1, stats = load_data(data_path, '01-01-2020', '01-01-2022')

series = find_series_lengths(stats['log_returns'].tolist())
sns.histplot(series, discrete=True)
plt.show()
