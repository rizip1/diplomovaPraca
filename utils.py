import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import pandas as pd


def save_autocorrect_state(model_errors, pred_length, x_train_sets,
                           x_test):
    '''
    Used for testing purpose to check if autocorrection
    errors are picked the right way.
    Will finish execution of program when finished.
    '''
    pd.Series(model_errors).to_csv('test/model_errors.csv')
    for i in range(pred_length):
        x_train_sets[i].to_csv('test/{}.csv'.format(i))
    x_test.to_csv('test/test.csv')
    import os
    os.sys.exit(1)


def contains_missing_data(x_train, y_train, x_test, y_test):
    missing_data_value = -999

    return (x_train.__contains__(missing_data_value) or
            y_train.__contains__(missing_data_value) or
            x_test.__contains__(missing_data_value) or
            y_test.__contains__(missing_data_value))


def can_use_autocorrect(model_errors, interval, window_len):
    return len(model_errors) > (interval * window_len) + 24


def get_autocorrect_col(model_errors, pos, interval, window_length):
    '''
    only for train data
    '''
    autocorrect_col = np.array([])

    for i in range(window_length):
        autocorrect_col = np.append(
            autocorrect_col, model_errors[-24 + pos - ((i + 1) * interval)])

    return np.flip(autocorrect_col, axis=0)


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
    real_values_x = [i for i in range(
        len(real_values) - (len(real_values) - len(predicted_values)))]

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot(predicted_values, 'or', label='Predicted values (Our model)')
    # plt.plot(shmu_predictions, 'og', label='Predicted values (SHMU)')
    plt.plot(real_values_x, real_values[-len(predicted_values):],
             'ok', label='Real values')
    plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
    plt.title('Temperature predictions')
    plt.ylabel('Temperature')
    plt.xlabel('Samples')
    plt.savefig('other/predictions.png')
    plt.close()


def save_errors(predicted_errors, shmu_errors, cum_mse, cum_mae):
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

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot(cum_mse, 'k', label='cummulative mse')
    plt.plot(cum_mae, 'r', label='cummulative mae')
    plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
    plt.title('Cummulative errors')
    plt.ylabel('Error')
    plt.xlabel('Samples')
    plt.savefig('other/cum_errors.png')
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
    parser.add_argument('--temperature-var', action='store',
                        default=0,
                        dest='temperature_var',
                        help='''Add temperature variance from
time when prediction was made and arg-1 hours before.\n''')
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
    parser.add_argument('--autocorrect', action='store_true',
                        dest='autocorrect',
                        default=False,
                        help='Use autocorrection')
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        default=False,
                        help='Verbose output')
    return parser


def predict(data, x, y, weight, models, window_len, interval=12, diff=False,
            norm=False, average_models=False, autocorrect=False,
            verbose=False):
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
    predictions_made = 0
    mae_predict = 0
    mse_predict = 0
    mae_shmu = 0
    mse_shmu = 0
    start = 0
    predicted_all = np.array([])
    train_end = window_len * interval
    model_errors = np.array([])
    model_predictions = np.array([])
    model_bias = 0
    cum_mse = []
    cum_mae = []

    while (train_end < data_len):
        if (verbose and predictions_made > 0):
            print('train_end', train_end, mae_predict /
                  predictions_made, mse_predict / predictions_made,
                  model_bias / predictions_made)

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

        autocorrect_ready = can_use_autocorrect(
            model_errors, interval, window_len)

        for i in range(pred_length):
            x_train = x_diff.iloc[start + i:train_end:interval, :]
            if (autocorrect and autocorrect_ready):
                autocorrect_col = get_autocorrect_col(model_errors, i,
                                                      interval, window_len)
                x_train['autocorrect'] = pd.Series(
                    autocorrect_col, index=x_train.index)
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
        if (autocorrect and autocorrect_ready):
            x_test['autocorrect'] = pd.Series(
                model_errors[-24:-24 + pred_length], index=x_test.index)

        x_test_orig = x_orig.iloc[train_end:train_end + pred_length, :]
        y_test = y_orig.iloc[train_end:train_end + pred_length]

        # for testing purpose
        '''
        if (predictions_made > 0):
            save_autocorrect_state(model_errors, pred_length, x_train_sets,
                               x_test)
        '''

        for i in range(pred_length):
            if (contains_missing_data(
                    x_train_sets[i].values, y_train_sets[i].values,
                    x_test.iloc[i].values, np.matrix(y_test.iloc[i]))):
                continue

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

            model_errors = np.append(
                model_errors, (y_predicted - y_test.iloc[i])[0])

            model_predictions = np.append(
                model_predictions, y_predicted)

            if (autocorrect and not autocorrect_ready):
                continue

            predictions_made += 1

            # add into predicted_all
            predicted_all = np.hstack((predicted_all, y_predicted))

            mae_shmu += np.sum(abs(y_test.iloc[i] -
                                   x_test_orig.future_temp_shmu.iloc[i]))
            mse_shmu += np.sum((y_test.iloc[i] -
                                x_test_orig.future_temp_shmu.iloc[i]) ** 2)

            mae_predict += np.sum(abs(y_test.iloc[i] - y_predicted))
            mse_predict += np.sum((y_test.iloc[i] - y_predicted) ** 2)
            model_bias += (y_test.iloc[i] - y_predicted)[0]

            cum_mse.append(mse_predict / predictions_made)
            cum_mae.append(mae_predict / predictions_made)

        # shift interval for learning
        train_end += pred_length
        start += pred_length

    return {
        'mae_predict': mae_predict / predictions_made,
        'mae_shmu': mae_shmu / predictions_made,
        'mse_predict': mse_predict / predictions_made,
        'mse_shmu': mse_shmu / predictions_made,
        'predicted_all': np.array(predicted_all),
        'predictions_count': predictions_made,
        'model_bias': model_bias / predictions_made,
        'cum_mse': cum_mse,
        'cum_mae': cum_mae,
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

        # check how many prediction we can make within same ref_date
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
