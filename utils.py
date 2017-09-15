import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import re
import pandas as pd


def init_model_errors(interval=24):
    model_errors = {}
    for i in range(interval):
        model_errors[i] = []
    return model_errors


def can_use_autoreg(model_errors, window_len):
    for key, value in model_errors.items():
        if (len(value) < window_len + 1):
            return False
    return True


def get_starting_hour(data, start):
    ref_date = data.loc[start, 'validity_date']
    m = re.search(
        r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
        ref_date)
    hour = int(m.group(1))
    return hour


def get_best_model(models, x_train, y_train, weights):
    best_model = None
    best_score = -float('inf')
    for m in models:
        score = -float('inf')
        if (weights is not None):
            m.fit(x_train, y_train, sample_weight=weights)
            score = m.score(x_train, y_train, sample_weight=weights)
        else:
            m.fit(x_train, y_train)
            score = m.score(x_train, y_train)
        if (score > best_score):
            best_score = score
            best_model = m
    return best_model


def get_avg_prediction(models, x_train, y_train, x_test):
    predicted = 0
    for m in models:
        m.fit(x_train, y_train)
        predicted += m.predict(x_test)
    return predicted / len(models)


def get_bias(real, predicted):
    return np.mean(real - predicted)


def save_predictions(real_values, predicted_values, shmu_predictions):
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot(predicted_values, 'or', label='Predicted values (Our model)')
    plt.plot(shmu_predictions, 'og', label='Predicted values (SHMU)')
    plt.plot(real_values, 'ok', label='Real values')
    plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
    plt.title('Temperature predictions')
    plt.ylabel('Temperature')
    plt.xlabel('Samples')
    plt.savefig('other/predictions.png')
    plt.close()


def save_errors(predicted_errors, shmu_errors):
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot(predicted_errors, 'k', label='predicted errors')
    # plt.plot(shmu_errors, 'r', label='shmu errors')
    plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
    plt.title('Temperature errors')
    plt.ylabel('Error')
    plt.xlabel('Samples')
    plt.savefig('other/errors.png')
    plt.close()


def get_parser():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--weight', action='store', dest='weight_coef',
                        help='''Weight coefficient. If none supplied, no
weights will be used.''', type=float)
    parser.add_argument('--file', action='store', dest='data_file',
                        required=True,
                        help='''Path to data file that will be loaded.''')
    parser.add_argument('--mode', action='store', dest='mode',
                        default='window',
                        choices=['window', 'extended-window', 'train-set'],
                        help="""Mode to use for predictions:\n
window = use sliding window
extended-window = use window that grows over time
train-set = learn from fixed length train set\n
Default length for window, extended-window and train-set is 60.
To override it set '--length' option.""")
    parser.add_argument('--length', action='store', dest='length',
                        default=60,
                        help='Length of window, extended-window or train-set.')
    parser.add_argument('--lags', action='store', dest='lags',
                        default=0,
                        help='Length of window, extended-window or train-set.')
    parser.add_argument('--no-intercept', action='store_true', default=False,
                        dest='no_intercept',
                        help='If set will not use bias term.')
    parser.add_argument('--model', action='store', dest='model',
                        default='reg',
                        choices=['reg', 'svr', 'rf', 'kn', 'nn'
                                 'ens', 'ens-linear', 'ens-ens'],
                        help="Model to use for predictions:\n")
    parser.add_argument('--shmu-error-p-time', action='store',
                        dest='shmu_error_p_time',
                        default='0:1:0',
                        help='''Will use shmu error from time when prediction
was made. First agr specifies lags count. For no lag set it equal to 1.
Second arg specifies lag distance in hours. Default is 1.
Third arg specifies exponent func, 0 means no exponent func. \n''')
    parser.add_argument('--feature-p-time', action='store',
                        dest='feature_p_time',
                        help='''Except input in format lag_count:lag_by:feature_name.
The supplied feature will be lagged by count hours from prediction time,
including each lag.\n''')
    parser.add_argument('--feature', action='store',
                        dest='feature',
                        help='''Except input in format lag_count:lag_by:feature_name.
The supplied feature will be lagged by count hours, including each lag.\n''')
    parser.add_argument('--temperature_var', action='store',
                        default=0,
                        dest='temperature_var',
                        help='''TODO.\n''')
    parser.add_argument('--shmu-error-var', action='store',
                        default=0,
                        dest='shmu_error_var',
                        help='''Add shmu error variance from
time when prediction was made and arg-1 hours before.\n''')
    parser.add_argument('--diff', action='store_true', dest='diff',
                        default=False,
                        help='Perform one step difference')
    parser.add_argument('--step', action='store', dest='step',
                        default=12,
                        help='Hour interval between learning examples')
    parser.add_argument('--norm', action='store_true', dest='norm',
                        default=False,
                        help='Normalize with mean and std')
    parser.add_argument('--avg', action='store_true', dest='average_models',
                        default=False,
                        help='Average models')
    parser.add_argument('--autoreg', action='store_true', dest='autoreg',
                        default=False,
                        help='Use autoregression')
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        default=False,
                        help='Verbose output')
    return parser


def predict(data, x, y, weight, models, window_len, interval=12, diff=False,
            norm=False, average_models=False, autoreg=False, verbose=False):
    '''
    Predict by looking at conditions that occured every `interval`
    hours earlier.
    The `window_len` determines length of sliding window.
    Can optionaly use `diff` argument to to perform one step difference
    on features.

    Base features are measured temperature every `window_len * interval`
    hours earlier and predicted temperature for `p` hours ahead every
    `window_len * interval` hours earlier.
    '''

    x_orig = None
    y_orig = None
    x_diff = None
    y_diff = None
    if (diff):
        x_orig = x.iloc[interval:]
        y_orig = y.iloc[interval:]
        x_diff = x.diff(periods=interval).iloc[interval:]
        y_diff = y.diff(periods=interval).iloc[interval:]
    else:
        x_orig = x
        y_orig = y
        x_diff = x
        y_diff = y

    data_len = x_diff.shape[0]
    predictions_count = data_len - (window_len * interval)
    mae_predict = 0
    mse_predict = 0
    mae_shmu = 0
    mse_shmu = 0
    start = 0
    predicted_all = np.array([])
    train_end = window_len * interval
    model_errors = init_model_errors(interval)
    model_bias = 0

    autoreg_ok = False

    while (train_end < data_len):
        if (verbose and len(predicted_all)):
            print('train_end', train_end, mae_predict /
                  len(predicted_all), mse_predict / len(predicted_all))
            # Check how many prediction we can make within same ref_date
        ref_date = data.reference_date[train_end]

        pred_length = 0
        while (data.reference_date[train_end + pred_length] == ref_date):
            pred_length += 1
            # Out of bounds
            if (pred_length + train_end >= data_len):
                break

        x_train_sets = []
        means = []
        stds = []
        y_train_sets_orig = []
        y_train_sets = []

        starting_hour = get_starting_hour(data, train_end)

        for i in range(pred_length):
            x_train = x_diff.iloc[start + i:train_end:interval, :]
            if (autoreg and (autoreg_ok or can_use_autoreg(model_errors,
                                                           window_len))):
                autoreg_ok = True
                current_hour = starting_hour + i
                if (current_hour != 0):
                    current_hour %= interval
                x_train['auto'] = pd.Series(
                    model_errors[current_hour][-window_len - 1:-1],
                    index=x_train.index)
            if (norm):
                std = x_train.std()
                mean = x_train.mean()
                means.append(mean)
                stds.append(std)
                x_train = (x_train - mean) / std

            x_train_sets.append(x_train)
            y_train_sets_orig.append(y_orig.iloc[start + i:train_end:interval])
            y_train_sets.append(y_diff.iloc[start + i:train_end:interval])

        x_test = x_diff.iloc[train_end:train_end + pred_length, :]
        if (autoreg and (autoreg_ok or can_use_autoreg(model_errors,
                                                       window_len))):
            to_add = []
            for i in range(pred_length):
                current_hour = starting_hour + i
                if (current_hour != 0):
                    current_hour %= interval
                to_add.append(model_errors[current_hour][-1])
            x_test['auto'] = pd.Series(to_add, index=x_test.index)

        x_test_orig = x_orig.iloc[train_end:train_end + pred_length, :]
        y_test = y_orig.iloc[train_end:train_end + pred_length]

        for i in range(pred_length):
            best_model = None
            weights = None
            y_predicted = 0
            if (weight):
                w = list(reversed([math.sqrt(weight ** j)
                                   for j in range(x_train_sets[i].shape[0])]))
                weights = np.array(w)

            x_test_item = x_test.iloc[i, :]
            if (norm):
                x_test_item = (x_test_item - means[i]) / stds[i]

            if (not average_models):
                best_model = get_best_model(
                    models, x_train_sets[i], y_train_sets[i], weights)
                y_predicted = best_model.predict(
                    x_test_item.values.reshape(1, -1))
            else:
                y_predicted = get_avg_prediction(
                    models, x_train_sets[i],
                    y_train_sets[i],
                    x_test_item.values.reshape(1, -1))

            if (diff):
                y_predicted += y_train_sets_orig[i].iloc[-1]

            # add into predicted_all
            predicted_all = np.hstack((predicted_all, y_predicted))

            mae_shmu += np.sum(abs(y_test.iloc[i] -
                                   x_test_orig.future_temp_shmu.iloc[i]))
            mse_shmu += np.sum((y_test.iloc[i] -
                                x_test_orig.future_temp_shmu.iloc[i]) ** 2)

            current_hour = starting_hour + i
            if (current_hour != 0):
                current_hour %= interval
            model_errors[current_hour].append(
                (y_test.iloc[i] - y_predicted)[0])

            mae_predict += np.sum(abs(y_test.iloc[i] - y_predicted))
            mse_predict += np.sum((y_test.iloc[i] - y_predicted) ** 2)
            model_bias += (y_test.iloc[i] - y_predicted)[0]

        # shift interval for learning
        train_end += pred_length
        start += pred_length

    return {
        'mae_predict': mae_predict / predictions_count,
        'mae_shmu': mae_shmu / predictions_count,
        'mse_predict': mse_predict / predictions_count,
        'mse_shmu': mse_shmu / predictions_count,
        'predicted_all': np.array(predicted_all),
        'predictions_count': predictions_count,
        'model_bias': model_bias / predictions_count,
    }


def predict_test(data, x, y, weight, models, length, step):
    data_len = x.shape[0]
    predictions_count = data_len - (length * step)
    mae_predict = 0
    mse_predict = 0
    mae_shmu = 0
    mse_shmu = 0
    start = 0
    predicted_all = np.array([])
    train_end = length * step
    model = models[0]

    while (train_end < data_len):
        x_train = x.iloc[start:train_end, :]
        y_train = y.iloc[start:train_end]

        # Check how many prediction we can make within same ref_date
        ref_date = data.reference_date[train_end]

        pred_length = 0
        while (data.reference_date[train_end + pred_length] == ref_date):
            pred_length += 1
            # Out of bounds
            if (pred_length + train_end >= data_len):
                break

        # test set if for 1 to 12 hours ahead
        # if dataset has ended it is 1 to x hours ahead
        # where x is in [1,12]
        x_test = x.iloc[train_end:train_end + pred_length, :]
        y_test = y.iloc[train_end:train_end + pred_length]

        weights = None
        if (weight):
            weights = list(reversed([math.sqrt(weight ** j)
                                     for j in range(x_train.shape[0])]))
            weights = np.array(weights)

        # values is not needed here, pandas removes Index by default
        model.fit(x_train.values, y_train, sample_weight=weights)

        # predict values for y
        y_predicted = model.predict(x_test)

        # add into predicted all
        predicted_all = np.hstack((predicted_all, y_predicted))

        # -1 index stands for current_temperature column in data
        mae_shmu += np.sum(abs(y_test - x_test.future_temp_shmu))
        mse_shmu += np.sum((y_test - x_test.future_temp_shmu) ** 2)

        mae_predict += np.sum(abs(y_test - y_predicted))
        mse_predict += np.sum((y_test - y_predicted) ** 2)

        # shift interval for learning
        train_end += pred_length
        start += pred_length

    return {
        'mae_predict': mae_predict / predictions_count,
        'mae_shmu': mae_shmu / predictions_count,
        'mse_predict': mse_predict / predictions_count,
        'mse_shmu': mse_shmu / predictions_count,
        'predicted_all': np.array(predicted_all),
        'predictions_count': predictions_count,
        'model_bias': 0,  # TODO
    }
