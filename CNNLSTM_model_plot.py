import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import math
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
import datetime
import time
import seaborn
from matplotlib import pyplot as plt
from data_processing import get_CMAPSSData, get_PHM08Data, data_augmentation, analyse_Data
from utils_laj import *
from numpy import save
today = datetime.date.today()
from data_processing import RESCALE, test_engine_id

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print('Using GPU', tf.config.list_physical_devices('GPU'))

def CNNLSTMModel(x_train, y_train, x_test, y_test, Train=True, folder_name='Raw Data', checkpoint='fold1'):
    '''
        The architecture is a Meny-to-meny model combining CNN and LSTM models
        :param dataset: select the specific dataset between PHM08 or CMAPSS
        :param Train: select between training and testing
        '''

    #### checkpoint saving path ####epoch
    #path_checkpoint = './Save/Save_CNNLSTM/'+folder_name+'/' + checkpoint
    path_checkpoint = './Save/CNNLSTM/' + folder_name + '/' + checkpoint

    ##################################

    batch_size = 6064  # Batch size 1024
    if Train == False: batch_size = 5

    sequence_length = 100  # Number of steps
    learning_rate = 0.001  # 0.0001
    epochs = 500
    ann_hidden = 50

    n_channels = 8 #28

    lstm_size = n_channels * 3  # 3 times the amount of channels
    num_layers = 2  # 2  # Number of layers

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, sequence_length, n_channels], name='inputs')
    Y = tf.placeholder(tf.float32, [None, sequence_length], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
    is_train = tf.placeholder(dtype=tf.bool, shape=None, name="is_train")


    conv1 = conv_layer(X, filters=18, kernel_size=2, strides=1, padding='same', batch_norm=False, is_train=is_train,
                       scope='conv_1')
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same', name='maxpool_1')

    conv2 = conv_layer(max_pool_1, filters=36, kernel_size=2, strides=1, padding='same', batch_norm=False,
                       is_train=is_train, scope='conv_2')
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same', name='maxpool_2')

    conv3 = conv_layer(max_pool_2, filters=72, kernel_size=2, strides=1, padding='same', batch_norm=False,
                       is_train=is_train, scope='conv_3')
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same', name='maxpool_3')

    conv_last_layer = max_pool_3

    shape = conv_last_layer.get_shape().as_list()
    CNN_flat = tf.reshape(conv_last_layer, [-1, shape[1] * shape[2]])

    dence_layer_1 = dense_layer(CNN_flat, size=sequence_length * n_channels, activation_fn=tf.nn.relu, batch_norm=False,
                                phase=is_train, drop_out=True, keep_prob=keep_prob,
                                scope="fc_1")
    lstm_input = tf.reshape(dence_layer_1, [-1, sequence_length, n_channels])

    cell = get_RNNCell(['LSTM'] * num_layers, keep_prob=keep_prob, state_size=lstm_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_output, states = tf.nn.dynamic_rnn(cell, lstm_input, dtype=tf.float32, initial_state=init_state)
    stacked_rnn_output = tf.reshape(rnn_output, [-1, lstm_size])  # change the form into a tensor

    dence_layer_2 = dense_layer(stacked_rnn_output, size=ann_hidden, activation_fn=tf.nn.relu, batch_norm=False,
                                phase=is_train, drop_out=True, keep_prob=keep_prob,
                                scope="fc_2")

    output = dense_layer(dence_layer_2, size=1, activation_fn=None, batch_norm=False, phase=is_train, drop_out=False,
                         keep_prob=keep_prob,
                         scope="fc_3_output")

    prediction = tf.reshape(output, [-1])
    y_flat = tf.reshape(Y, [-1])

    h = prediction - y_flat

    cost_function = tf.reduce_sum(tf.square(h))
    RMSE = tf.sqrt(tf.reduce_mean(tf.square(h)))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost_function)

    saver = tf.train.Saver()
    training_generator = batch_generator(x_train, y_train, batch_size, sequence_length, online=True)
    testing_generator = batch_generator(x_test, y_test, batch_size, sequence_length, online=False)

    if Train: model_summary(learning_rate=learning_rate, batch_size=batch_size, lstm_layers=num_layers,
                            lstm_layer_size=lstm_size, fc_layer_size=ann_hidden, sequence_length=sequence_length,
                            n_channels=n_channels, path_checkpoint=path_checkpoint, spacial_note='')

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        if Train == True:
            cost = []
            iteration = int(x_train.shape[0] / batch_size)
            print("Training set MSE")
            print("No epoches: ", epochs, "No itr: ", iteration)
            __start = time.time()
            for ep in range(epochs):

                for itr in range(iteration):
                    ## training ##
                    batch_x, batch_y = next(training_generator)
                    session.run(optimizer,
                                feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8, learning_rate_: learning_rate})
                    cost.append(
                        RMSE.eval(feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0, learning_rate_: learning_rate}))

                x_test_batch, y_test_batch = next(testing_generator)
                mse_train, rmse_train = session.run([cost_function, RMSE],
                                                    feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0,
                                                               learning_rate_: learning_rate})
                mse_test, rmse_test = session.run([cost_function, RMSE],
                                                  feed_dict={X: x_test_batch, Y: y_test_batch, keep_prob: 1.0,
                                                             learning_rate_: learning_rate})

                time_per_ep = (time.time() - __start)
                time_remaining = ((epochs - ep) * time_per_ep) / 3600
                print("CNNLSTM", "epoch:", ep, "\tTrainig-",
                      "MSE:", mse_train, "RMSE:", rmse_train, "\tTesting-", "MSE", mse_test, "RMSE", rmse_test,
                      "\ttime/epoch:", round(time_per_ep, 2), "\ttime_remaining: ",
                      int(time_remaining), " hr:", round((time_remaining % 1) * 60, 1), " min", "\ttime_stamp: ",
                      datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
                __start = time.time()

                if ep % 10 == 0 and ep != 0:
                    save_path = saver.save(session, path_checkpoint)
                    if os.path.exists(path_checkpoint + '.meta'):
                        print("Model saved to file: %s" % path_checkpoint)
                    else:
                        print("NOT SAVED!!!", path_checkpoint)

                if ep % 1000 == 0 and ep != 0: learning_rate = learning_rate / 10

            save_path = saver.save(session, path_checkpoint)
            if os.path.exists(path_checkpoint + '.meta'):
                print("Model saved to file: %s" % path_checkpoint)
            else:
                print("NOT SAVED!!!", path_checkpoint)
            #plt.plot(cost)
            #plt.show()
        else:
            saver.restore(session, path_checkpoint)
            print("Model restored from file: %s" % path_checkpoint)

            x_validation = x_test
            y_validation = y_test

            validation_generator = batch_generator(x_validation, y_validation, batch_size, sequence_length,
                                                   online=True,
                                                   online_shift=sequence_length)

            full_prediction = []
            actual_rul = []
            error_list = []

            iteration = int(x_validation.shape[0] / (batch_size * sequence_length))
            print("#of validation points:", x_validation.shape[0], "#datapoints covers from minibatch:",
                  batch_size * sequence_length, "iterations/epoch", iteration)

            for itr in range(iteration):
                x_validate_batch, y_validate_batch = next(validation_generator)
                __y_pred, error, __y = session.run([prediction, h, y_flat],
                                                   feed_dict={X: x_validate_batch, Y: y_validate_batch,
                                                              keep_prob: 1.0})
                full_prediction.append(__y_pred * RESCALE)
                actual_rul.append(__y * RESCALE)
                error_list.append(error * RESCALE)
            full_prediction = np.array(full_prediction)
            full_prediction = full_prediction.ravel()
            actual_rul = np.array(actual_rul)
            actual_rul = actual_rul.ravel()
            error_list = np.array(error_list)
            error_list = error_list.ravel()
            rmse = np.sqrt(np.sum(np.square(error_list)) / len(error_list))  # RMSE
            mae = np.average(np.abs(error_list))  # MAE
            r2 = ((np.corrcoef(full_prediction, actual_rul))[0, 1]) ** 2

            print(y_validation.shape, full_prediction.shape, full_prediction, actual_rul, prediction, "RMSE:", rmse,
                  "MAE:",
                  mae, "Score:", scoring_func(error_list), "R2:", r2)
            return full_prediction, actual_rul, error_list



def plot_alpha_lambda_acc(all_predictions):
    import pylab as plt

    engine_ids_plot = [56, 48, 19,  9, 54, 41, 91, 39, 58]
    for index_engine_id in range(10):
        engineID = engine_ids_plot[index_engine_id]

        params = {'text.usetex': True,
                  'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
        SMALL_SIZE = 12
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 18
        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.rcParams.update(params)

        CB91_Blue = '#2CBDFE'
        CB91_Green = '#47DBCD'
        CB91_Pink = '#F3A0F2'
        CB91_Purple = '#9D2EC5'
        CB91_Violet = '#661D98'
        CB91_Amber = '#F5B14C'
        CB91_Black = '#000000'

        colors = [CB91_Blue, CB91_Green, CB91_Purple, CB91_Amber, CB91_Pink, CB91_Black]


        # And a corresponding grid

        plt.gca().grid(which='both', linestyle='--')
        print(all_predictions)
        for predictions, test_ids, test_labels, method_name, color in zip(all_predictions, all_test_ids, all_test_labels, method_names, colors):
            print("engine", engineID, np.unique(test_ids), test_ids, test_labels)
            max_RUL_engineID = np.max(test_labels[test_ids == engineID])
            predictions = predictions[test_ids == engineID]
            if color == CB91_Blue:
                max_RUL = max_RUL_engineID
                plt.plot([max_RUL, 0], [max_RUL, 0], c='black')
                plt.gca().fill_between([0, max_RUL, max_RUL, 0], [0, max_RUL*1.2, max_RUL*0.8, 0], 1, color='gray', alpha=0.3,)
            plt.scatter(range(len(predictions[5:-5]), 0, -1), predictions[5:-5], label=method_name, c=color)
        plt.legend()
        plt.title('Engine ' + str(engineID))
        plt.xlabel('Time to EoL (cycles)')
        plt.ylabel('Remaining Useful Life (RUL)')
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.savefig('Pictures/results/RF' + str(engineID) + '.png')
        plt.close()
        #plt.show()


names = ['PreprocessedData\\orig.csv', 'PreprocessedData\\sma.csv', 'PreprocessedData\\ema01.csv', 'PreprocessedData\\ema02.csv', 'PreprocessedData\\ema03.csv',
         'PreprocessedData\\acd.csv']
method_names = ['Raw data', 'SMA', 'EMA0.1', 'EMA0.2', 'EMA0.3', 'ACD']


all_data = []

for name in names:
    data = pd.read_csv(name)
    all_data.append(data)


feature_list = []
for sensorid in [2, 3, 4, 8, 11, 13, 17]: # [2, 3 , 4 , 7 , 8 , 9 , 11, 12 , 13 , 14 , 15 , 17 , 20 , 21]:
    feature_list.append('sensor' + str(sensorid))
feature_list.append('cycle')

for data in all_data:
    for feature in feature_list:
        if feature == 'cycle': continue
        data[feature] = (data[feature] - np.mean(data[feature]))/ np.std(data[feature])

    feature = 'cycle'
    mean_time = np.mean(data[feature])
    stdev_time = np.std(data[feature])
    data[feature] = (data[feature] - np.mean(data[feature])) / np.std(data[feature])

    feature = 'RUL'
    mean_RUL  = np.mean(data[feature])
    stdev_RUL = np.std(data[feature])
    data[feature] = (data[feature] - np.mean(data[feature]))/ np.std(data[feature])

all_datasets_mae = []
all_datasets_rmse = []
all_datasets_score = []
all_datasets_alpha_lambda = []
all_predictions, all_test_ids = [], []
all_test_labels = []
for i in range(10):
    all_datasets_alpha_lambda.append([])
for i in range(len(method_names)):
    all_predictions.append([])
    all_test_ids.append([])
    all_test_labels.append([])


uniqueIds = data['id'].unique()
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42,
                           max_depth=8, min_samples_leaf=50)

for data, method_name, index_method in zip(all_data, method_names, range(len(method_names))):
    print(method_name)
    score_all = []
    mae_all = []
    rmse_all = []
    accuracy_all = []
    for i in range(0,11):
        sublist = []
        accuracy_all.append(sublist)

    cross_i = 1

    uniqueIds = data['id'].unique()

    print(len(uniqueIds), uniqueIds)
    np.random.shuffle(uniqueIds)
    uniqueIds = [ 56, 48, 19,  9, 54, 41, 91, 39, 58, 79, 25,  5, 27, 53, 36, 20, 15, 42
    ,  7, 80,  3, 83, 40, 82, 77, 51, 62, 16, 93, 81, 85, 92, 98, 66, 89, 65
    , 57,  4, 34, 52, 35, 88, 86,  8, 69, 75, 33, 13, 72, 29, 30, 99, 70, 50
    , 38, 31, 55, 23, 43, 21, 94, 64, 96, 12, 22, 37, 84, 18, 11, 14, 97, 60
    , 61, 68, 10, 90, 26, 76, 28, 45, 59, 100, 32,  1, 49, 63, 71, 17,  2, 67
    , 46, 95, 24, 73,  6, 78, 74, 47, 44, 87]

    for i in range(0, 10, 10):
        test_ids = uniqueIds[i:i+10]
        print('test ids:', test_ids)
        train_ids = [i for i in uniqueIds if i not in test_ids]
        train_data = data.loc[data['id'].isin(train_ids), feature_list]
        train_labels = data.loc[data['id'].isin(train_ids), 'RUL'].values
        test_data = data.loc[data['id'].isin(test_ids), feature_list]
        test_labels = data.loc[data['id'].isin(test_ids), 'RUL'].values
        test_times = data.loc[data['id'].isin(test_ids), 'cycle'].values
        test_identifiers = data.loc[data['id'].isin(test_ids), 'id'].values

        if True:
            # Train the model on training data
            rf.fit(train_data, train_labels);
            # Use the forest's predict method on the test data
            predictions = rf.predict(test_data)
            predictions = (predictions * stdev_RUL) + mean_RUL
            test_labels = (test_labels * stdev_RUL) + mean_RUL
            test_times = (test_times * stdev_time) + mean_time

        if False:
            CNNLSTMModel(train_data, train_labels, test_data, test_labels, Train=True, folder_name=method_name,
                         checkpoint='fold' + str(cross_i))
            cross_i += 1
            continue

        if False:
            predictions, test_labels, error_list = CNNLSTMModel(train_data, train_labels, test_data, test_labels, Train=False,  folder_name=method_name, checkpoint='fold' + str(cross_i))
            predictions = (predictions * stdev_RUL) + mean_RUL
            test_labels = (test_labels * stdev_RUL) + mean_RUL
            error_list = (error_list * stdev_RUL) + mean_RUL
            test_times = (test_times * stdev_time) + mean_time

        test_times = test_times[:len(test_labels)]
        all_test_labels[index_method] = test_labels[:len(test_labels)]
        all_test_ids[index_method] = test_identifiers[:len(test_labels)]
        all_predictions[index_method] = predictions
        continue



        if False:
            id_clause = test_ids == test_ids[0]
            plt.plot(test_times[test_ids == test_ids[0]], test_times[test_ids == test_ids[0]])
            plt.scatter(test_times[test_ids == test_ids[0]], predictions[id_clause])
            plt.scatter(test_times[test_ids == test_ids[0]], predictions[id_clause], label='ACD')
            plt.show()

        cross_i += 1
        #continue


        # Calculate the absolute errors
        errors = predictions - test_labels
        if False:
            plt.scatter(range(500), all_data[0].loc[data['id'].isin(test_ids), feature_list]['sensor3'][:500], label='orig')
            plt.scatter(range(500), test_data['sensor3'][:500])
            plt.scatter(range(500), train_data['sensor3'][:500], c='red')
            #plt.plot(errors)
            plt.legend()
            plt.show()

        scores = []
        for error in errors:
            if error < 0:
                scores.append(math.exp(-error / 10) - 1)
            else:
                scores.append(math.exp(-error / 13) - 1)

        score = np.mean(scores)
        mae = np.mean(np.abs(errors))
        rmse = math.sqrt(np.mean(np.square(errors)))

        accuracy = [0]*11
        counts = [0]*11

        for time, predicted_rul, rul in zip(test_times, predictions, test_labels):
            lambda_ = int((time / (time + rul)) * 10)
            if predicted_rul <= rul * 1.3 and predicted_rul >= rul * 0.7:
                accuracy[lambda_] += 1
            counts[lambda_] += 1

        for acc, count, lambda_ in zip(accuracy, counts, range(10)):
            print('Accuracy at ', lambda_, acc/count)

        for lambda_ in range(10):
            accuracy_all[lambda_].append(accuracy[lambda_] / counts[lambda_])
        print(cross_i, 'Cross validation')

            # Print out the metrics
        print('Mean Absolute Error:', mae, 'cycles.')
        print('Root Mean Squared Error:', rmse)
        print('Scoring function PHM08:', score)

        score_all.append(score)
        rmse_all.append(rmse)
        mae_all.append(mae)


    print(method_name)
    continue

    print('Mean Absolute Error:', np.round(np.mean(mae_all),2), '$\pm$',  np.round(np.std(mae_all),2), ' & ', sep='')
    print('Root Mean Squared Error:', np.round(np.mean(rmse),2), '$\pm$',  np.round(np.std(rmse_all),2), ' & ', sep='')
    print('Scoring function PHM08:', np.round(np.mean(score_all),2), '$\pm$',  np.round(np.std(score_all),2),'& ',  sep='')


    all_datasets_mae.append(str(np.round(np.mean(mae_all),2)) + '$\pm$'  +str(np.round(np.std(mae_all),2)))
    all_datasets_rmse.append(str(np.round(np.mean(rmse_all), 2)) + '$\pm$' + str(np.round(np.std(rmse_all), 2)))
    all_datasets_score.append(str(np.round(np.mean(score_all), 2)))

    for lambda_ in range(10):
        acc_alphalambda = str(np.round(np.mean(accuracy_all[lambda_]) * 100, 2)) + '$\pm$' + str(np.round(np.std(accuracy_all[lambda_]) * 100, 2))
        all_datasets_alpha_lambda[lambda_].append(acc_alphalambda)
        print('alpha lambda acc', lambda_, ' ', np.round(np.mean(accuracy_all[lambda_])*100,2),' & ', sep='')

plot_alpha_lambda_acc(all_predictions)

print('MAE &')
for mae in all_datasets_mae:
    print(mae + '&', sep='')
print('\\\\')
print('RMSE &')
for rmse in all_datasets_rmse:
    print(rmse + '&', sep='')
print('\\\\')
print('Score08 &')
for score in all_datasets_score:
    print('{:.2e}'.format(score),'&', sep='')
print('\\\\')

for lambda_ in range(10):
    print('$\\alpha\mbox{-}\lambda\\text{ }' + str((lambda_ + 1) * 10) + '\%$ &')
    for alphalambdaacc in all_datasets_alpha_lambda[lambda_]:
        print(alphalambdaacc + '&', sep='')
    print('\\\\')
