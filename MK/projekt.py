#from importlib.metadata import pass_none
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib.dates import DateFormatter, MonthLocator
from math import log, e
from data_loader import load_data
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data
from simulation import generate_gbm_paths, bootstrap_paths


data_path = os.path.join('..', 'data', 'wig20_d_full.csv')
data, stats = load_data(data_path)
data['Date'] = pd.to_datetime(data['Date']).dt.date
print(data.columns)
#print(data.describe())
#print(data.head())







