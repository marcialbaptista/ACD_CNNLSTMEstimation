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
            rolmean_signal = result_train.loc[result_train['id'] == engineID, 'sensor' + str(i+1)].rolling(window_size, min_periods=1).mean()
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
    for engineID in uniqueIds:
        for i in range(21):
            orig_signal = result_train.loc[result_train['id'] == engineID, 'sensor' + str(i+1)]
            if i+1 not in [1, 5, 6, 10, 16, 18, 19]:
                mon_signal = acd_algorithm.increase_monotonicity(orig_signal)
                result_acd.loc[result_acd.id == engineID, 'sensor' + str(i+1)] = mon_signal
    return result_acd

result_rolmean = rolling_mean(11)
result_acd = acd()
results = [result_train,  result_rolmean, ema(0.1), ema(0.2), ema(0.3), result_acd]
names = ['PreprocessedData\\orig.csv', 'PreprocessedData\\sma.csv', 'PreprocessedData\\ema01.csv', 'PreprocessedData\\ema02.csv', 'PreprocessedData\\ema03.csv',
         'PreprocessedData\\acd.csv']

for result, name in zip(results, names):
    result.to_csv(name, index=True)