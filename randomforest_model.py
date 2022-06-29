import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import math
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor


names = ['PreprocessedData\\orig.csv', 'PreprocessedData\\sma.csv', 'PreprocessedData\\ema01.csv', 'PreprocessedData\\ema02.csv', 'PreprocessedData\\ema03.csv',
         'PreprocessedData\\acd.csv']
methods_names = ['Raw data', 'SMA', 'EMA0.1', 'EMA0.2', 'EMA0.3', 'ACD']
all_data = []

for name in names:
    data = pd.read_csv(name)
    all_data.append(data)

data = all_data[0]

feature_list = []
for sensorid in [2, 3 , 4 , 7 , 8 , 9 , 11 , 12 , 13 , 14 , 15 , 17 , 20 , 21]:
    feature_list.append('sensor' + str(sensorid))
    feature_list.append('cycle')


score_all = []
mae_all = []
rmse_all = []
accuracy_all = []
for i in range(0,11):
    sublist = []
    accuracy_all.append(sublist)

cross_i = 0

uniqueIds = data['id'].unique()
print(len(uniqueIds), uniqueIds)
np.random.shuffle(uniqueIds)
for i in range(0, 100, 10):
    train_ids = uniqueIds[i:i+10]
    print(train_ids, i, i + 10)
    test_ids = [i for i in uniqueIds if i not in train_ids]
    train_data = data.loc[data['id'].isin(train_ids), feature_list]
    test_data = data.loc[data['id'].isin(test_ids), feature_list]
    train_labels = data.loc[data['id'].isin(train_ids), 'RUL']
    test_labels = data.loc[data['id'].isin(test_ids), 'RUL']
    test_times = data.loc[data['id'].isin(test_ids), 'cycle']

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    # Train the model on training data
    rf.fit(train_data, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_data)
    # Calculate the absolute errors
    errors = test_labels - predictions

    scores = []
    for error in errors:
        if error < 0:
            scores.append(math.exp(-error / 13) - 1)
        else:
            scores.append(math.exp(error / 10) - 1)

    score = np.mean(scores)
    mae = math.sqrt(np.mean(np.abs(errors)))
    rmse = math.sqrt(np.mean(np.square(errors)))

    accuracy = [0]*11
    counts = [0]*11

    for time, predicted_rul, rul in zip(test_times, predictions, test_labels):
        lambda_ = int((time / (time + rul)) * 10)
        if predicted_rul <= rul * 1.3 and predicted_rul >= rul * 0.7:
            accuracy[lambda_] += 1
        counts[lambda_] += 1

    for acc, count, lambda_ in zip(accuracy, counts, range(11)):
        print('Accuracy at ', lambda_, acc/count)

    for lambda_ in range(11):
        accuracy_all[lambda_].append(accuracy[lambda_] / counts[lambda_])
    print(cross_i, 'Cross validation')
    cross_i += 1

        # Print out the metrics
    print('Mean Absolute Error:', mae, 'cycles.')
    print('Root Mean Squared Error:', rmse)
    print('Scoring function PHM08:', score)

    score_all.append(score)
    rmse_all.append(rmse)
    mae_all.append(mae)

print('Mean Absolute Error:', np.mean(mae_all), np.std(mae_all))
print('Root Mean Squared Error:', np.mean(rmse), np.std(rmse_all))
print('Scoring function PHM08:', np.mean(score), np.std(score))
print(accuracy_all[0])
print(accuracy_all[1])
for lambda_ in range(11):
    print('alpha lambda acc', lambda_, np.mean(accuracy_all[lambda_]), np.std(accuracy_all[lambda_]))