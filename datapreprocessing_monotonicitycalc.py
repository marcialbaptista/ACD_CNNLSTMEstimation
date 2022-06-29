import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import math


names = ['PreprocessedData\\orig.csv', 'PreprocessedData\\sma.csv', 'PreprocessedData\\ema01.csv', 'PreprocessedData\\ema02.csv', 'PreprocessedData\\ema03.csv',
         'PreprocessedData\\acd.csv']
methods_names = ['Raw data', 'SMA', 'EMA0.1', 'EMA0.2', 'EMA0.3', 'ACD']
results = []

for name in names:
    data = pd.read_csv(name)
    results.append(data)

def calculate_coble_monotonicity(data):
    uniqueIds = data['id'].unique()
    total_monotonicity = []
    for i in range(21):
        mon_sensor = []
        if i+1 not in [1, 5, 6, 10, 16, 18, 19]:
            for engineID in uniqueIds:
                signal = data.loc[data['id'] == engineID, 'sensor' + str(i+1)]
                diff_signal = np.diff(signal)
                sum_signals = abs(np.sum(diff_signal < 0) - np.sum(diff_signal >= 0))
                mon_sensor_engine = sum_signals / len(signal.values)
                mon_sensor.append(np.abs(mon_sensor_engine))
                total_monotonicity.append(np.abs(mon_sensor_engine))
            print(round(np.mean(np.abs(mon_sensor)),2), '&', end = '')
    print(round(np.mean(total_monotonicity), 2),'$\pm$', round(np.std(total_monotonicity), 2), '\\', sep='')

def calculate_spearman_monotonicity(data):
    total_spearman = []
    uniqueIds = data['id'].unique()
    for i in range(21):
        spearman_sensor = []
        if i+1 not in [1, 5, 6, 10, 16, 18, 19]:
            for engineID in uniqueIds:
                signal = data.loc[data['id'] == engineID, 'sensor' + str(i+1)].values
                time = data.loc[data['id'] == engineID, 'cycle'].values
                coef, p = spearmanr(time, signal)
                if np.all(signal == signal[0]): # constant value array
                    continue
                total_spearman.append(abs(coef))
                spearman_sensor.append(abs(coef))
            print(round(np.mean(spearman_sensor), 2), '&', end='', sep='')
    print(round(np.nanmean(total_spearman),2),'$\pm$', round(np.nanstd(total_spearman),2), '\\', sep='')



for result, name, method_name in zip(results, names, methods_names):
    print(method_name, '&', sep='', end='')
    calculate_coble_monotonicity(result)

    #print(method_name, '&', sep='', end='')
    print(' ', '&', sep='', end='')
    calculate_spearman_monotonicity(result)