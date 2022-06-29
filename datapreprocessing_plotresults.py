import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
from ACD_proxy import ACD


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
    fileName = dataPath + '/FD00' + str(setNumber) + '_train.txt'
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
result_train, trainLifeCycles = loadTrainData(setNumber, decrease_threshold)
#plotSensorDataOfId(train, 1)

# Rolling mean application


def rolling_mean(window_size):
    uniqueIds = result_train['id'].unique()
    result_rolmean = result_train.copy()
    for engineID in uniqueIds:
        for i in range(21):
            rolmean_signal = result_train.loc[result_train['id'] == engineID, 'sensor' + str(i+1)].rolling(window_size).mean()
            result_rolmean.loc[result_rolmean.id == engineID, 'sensor' + str(i+1)] = rolmean_signal
    return result_rolmean

def ema(alpha):
    uniqueIds = result_train['id'].unique()
    result_ema = result_train.copy()
    for engineID in uniqueIds:
        for i in range(21):
            ema_signal = result_train.loc[result_train['id'] == engineID, 'sensor' + str(i+1)].ewm(alpha = alpha).mean()
            result_ema.loc[result_ema.id == engineID, 'sensor' + str(i+1)] = ema_signal
    return result_ema

mcId = 1
sensorID = 2
def acd():
    acd_algorithm = ACD()
    uniqueIds = result_train['id'].unique()
    result_acd = result_train.copy()
    for engineID in [mcId]: #uniqueIds:
        for i in [sensorID]:#range(21):
            orig_signal = result_train.loc[result_train['id'] == engineID, 'sensor' + str(i+1)]
            if i+1 not in [1, 5, 6, 10, 16, 18, 19]:
                mon_signal = acd_algorithm.increase_monotonicity(orig_signal)
                result_acd.loc[result_acd.id == engineID, 'sensor' + str(i+1)] = mon_signal
                print('Sensor', 'sensor' + str(i+1), 'engine', mcId)
                plt.plot(result_acd.loc[result_acd.id == engineID, 'sensor' + str(i+1)].values)
                plt.show()
    return result_acd

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

result_rolmean = rolling_mean(11)
result_acd = acd()
results = [result_train,  result_rolmean, ema(0.1), ema(0.2), ema(0.3), result_acd]
colors = [CB91_Blue, CB91_Green, CB91_Purple, CB91_Amber, CB91_Pink, CB91_Violet]

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

# And a corresponding grid
plt.gca().grid(which='both')

plt.figure(figsize=(2, 2))
names = ['Raw data', 'SMA', 'EMA', 'EMA', 'EMA', 'ACD']
ema_config = [0.1, 0.2, 0.3]
mcId = 1
sensorID = 2
i = 0
plot_i = 0
while i < len(names):
    name = names[i]
    sensor = 'sensor' + str(sensorID+1)
    plt.subplot(2, 2, plot_i + 1).set_title(name)

    if name == 'EMA':
        config_i = 0
        for j in range(3):
            color = colors[i]
            data = results[i]
            data = data[data['id'] == mcId]
            plt.plot(data['cycle'], data[sensor], c=color, label=str(ema_config[config_i]))
            config_i += 1
            i += 1
            plt.legend()
    else:
        color = colors[i]
        result_data = results[i]
        data = result_data[result_data['id'] == mcId]
        plt.plot(data['cycle'], data[sensor], c=color)
        i = i + 1
    plt.xlabel('Time')
    plt.ylabel('Sensor signal')
    plt.grid(which='both', alpha=0.5)
    plot_i += 1


# Or if you want different settings for the grids:
plt.tight_layout()
plt.show()