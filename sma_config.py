import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns


id_col = ['id']
cycle_col = ['cycle']
setting_cols = ['setting1', 'setting2', 'setting3']
sensor_cols = ['sensor' + str(i) for i in range(1, 22)]
rul_col = ['RUL']
all_cols = id_col + cycle_col + setting_cols + sensor_cols + rul_col

def loadData(fileName):
    data = pd.read_csv(fileName, sep=" ", header=None)
    #data.drop([26, 27], axis = 1, inplace=True)
    data.columns = id_col + cycle_col + setting_cols +sensor_cols
    return data


def addTrainRul(data, decrease_threshold=None):
    lifeCycles = {mcId: data[data['id'] == mcId]['cycle'].max() for mcId in data['id'].unique()}
    if decrease_threshold == None: decrease_threshold = 1
    ruls = [lifeCycles[row[0]] - decrease_threshold if row[1] < decrease_threshold else lifeCycles[row[0]] - row[1] for
            row in data.values]
    data['RUL'] = ruls
    return lifeCycles


# use this last one only, return the data as well as the max life cycles
def loadTrainData(setNumber, decrease_threshold=None):
    dataPath = 'CMAPSSData'
    fileName = dataPath + '/train_FD00' + str(setNumber) + '.txt'
    data = loadData(fileName)
    lifeCycles = addTrainRul(data, decrease_threshold)
    return data, lifeCycles


# As of feature selection they often select: 7, 8, 9, 12, 16, 17, 20  (manual selection based on sensor trends)
def plotSensorDataOfId(data, mcId):
    plt.figure(figsize=(30, 20))
    for i in range(21):
        sensor = 'sensor'+str(i+1)
        plt.subplot(10, 3, i+1).set_title(sensor)
        ssdata = data[data['id']==mcId]
        plt.plot(ssdata['cycle'], ssdata[sensor])
    plt.tight_layout()
    plt.show()

# train = raw data
setNumber = 1
decrease_threshold = None
train, trainLifeCycles = loadTrainData(setNumber, decrease_threshold)
#plotSensorDataOfId(train, 1)

# Rolling mean application
uniqueIds = train['id'].unique()

def rolling_mean(window_size):
    result_rolmean = train.copy()
    for engineID in uniqueIds:
        for i in range(21):
            rolmean_signal = train.loc[train['id'] == engineID, 'sensor' + str(i+1)].rolling(window_size).mean()
            result_rolmean.loc[result_rolmean.id == engineID, 'sensor' + str(i+1)] = rolmean_signal
    return result_rolmean

def calculate_sad_rolling_mean(window_size):
    result_rolmean = train.copy()
    points = 0
    for engineID in uniqueIds:
        for i in range(21):
            orig_signal = train.loc[train['id'] == engineID, 'sensor' + str(i+1)].values
            rolmean_signal = train.loc[train['id'] == engineID, 'sensor' + str(i+1)].rolling(window_size).mean()
            result_rolmean.loc[result_rolmean.id == engineID, 'sensor' + str(i+1)] = rolmean_signal
            # sum of absolute differences
            min_len = int(min(len(rolmean_signal), len(orig_signal)))
            sad = np.sum(np.abs(np.subtract(rolmean_signal[:min_len], orig_signal[:min_len])))
            points += min_len
    return sad

CB91_Blue = '#2CBDFE'

result_rolmean = rolling_mean(10)
#plotSensorDataOfId(result_rolmean, 1)

sad_list = []
min_window_size = 3
max_window_size = 40
for k in range(min_window_size, max_window_size):
    sad = calculate_sad_rolling_mean(k)
    sad_list.append(sad)

import pylab as plt
params = {'text.usetex': True,
          'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams.update(params)

major_ticks = np.arange(0, max_window_size, 5)
minor_ticks = np.arange(0, max_window_size, 1)

plt.gca().set_xticks(major_ticks)
plt.gca().set_xticks(minor_ticks, minor=True)

major_ticks = np.arange(8, 10, 0.1)
minor_ticks = np.arange(8, 10, 0.1)

plt.gca().set_yticks(major_ticks)
plt.gca().set_yticks(minor_ticks, minor=True)

# And a corresponding grid
plt.gca().grid(which='both')

# Or if you want different settings for the grids:
plt.gca().grid(which='minor', alpha=0.2)
plt.gca().grid(which='major', alpha=0.5)
plt.plot(range(min_window_size, max_window_size), sad_list, c=CB91_Blue)
plt.xlabel('Window size (w)')
plt.grid()


label = 'Optimal window size'
plt.annotate(label,
             xy=(11,sad_list[11 - min_window_size]),
             xycoords='data',
             xytext=(11,sad_list[11 - min_window_size]-0.1),
             textcoords='data',
             ha='center', arrowprops=dict(arrowstyle="->", color='black')) # horizontal alignment can be left, right or center

plt.ylabel('Sum of absolute differences')
plt.show()