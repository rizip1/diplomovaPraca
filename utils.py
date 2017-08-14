import matplotlib.pyplot as plt
import numpy as np
import argparse
import math


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
    plt.plot(shmu_errors, 'r', label='shmu errors')
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
                        dest='intercept',
                        help='If set will not use bias term.')
    parser.add_argument('--model', action='store', dest='model',
                        default='reg',
                        choices=['reg', 'svr', 'rf'],
                        help="Model to use for predictions:\n")
    parser.add_argument('--temp-day', action='store', dest='temp_day_lag',
                        default=0,
                        help='''Number of previous temperatures
lagged by 24 hours:\n''')
    parser.add_argument('--temp-hour', action='store', dest='temp_hour_lag',
                        default=0,
                        help='''Number of previous temperatures
lagged by 1 hour:\n''')
    return parser


def predict(data, x, y, weight, model, window_len, lags):
    '''
    Predict by looking window_len timestams before.
    Not sufficient enought better to use 12 * window_len
    predictions.
    '''
    data_len = x.shape[0]
    predictions_count = data_len - window_len
    mae_predict = 0
    mse_predict = 0
    mae_shmu = 0
    mse_shmu = 0
    start = 0
    predicted_all = np.array([])
    train_end = window_len

    model_errors = [0] * lags

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

        if (lags):
            if (len(model_errors) - 12 - lags >= window_len):
                for l in range(lags):
                    column_name = 'lags_{}'.format(l)

                    autoreg = np.array(
                        model_errors[-(window_len + 12 + l):-(12 + l)])
                    kwargs = {column_name: autoreg}
                    x_train = x_train.assign(**kwargs)

                    autoreg = np.array(model_errors[-(pred_length):])
                    kwargs = {column_name: autoreg}
                    x_test = x_test.assign(**kwargs)

        weights = None
        if (weight):
            weights = list(reversed([math.sqrt(weight ** j)
                                     for j in range(x_train.shape[0])]))
            weights = np.array(weights)

        # values is not needed here, pandas removes Index by default
        model.fit(x_train.values, y_train)

        # predict values for y
        y_predicted = model.predict(x_test)

        # add into predicted all
        predicted_all = np.hstack((predicted_all, y_predicted))

        # -1 index stands for current_temperature column in data
        mae_shmu += np.sum(abs(y_test - x_test.future_temp_shmu))
        mse_shmu += np.sum((y_test - x_test.future_temp_shmu) ** 2)

        mae_predict += np.sum(abs(y_test - y_predicted))
        mse_predict += np.sum((y_test - y_predicted) ** 2)
        model_errors += list(y_test - y_predicted)

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
    }


def predict_12_window(data, x, y, weight, model, window_len, lags):
    '''
    Predict by looking at conditions that happened every 12 hours earlier.
    window_len determines length of sliding window.

    Base features are measured temperature every window_len * 12 hours
    earlier and predicted temperature for p hours ahead every
    window_len * 12 hours earlier.
    '''
    interval = 12

    data_len = x.shape[0]
    predictions_count = data_len - (window_len * interval)
    mae_predict = 0
    mse_predict = 0
    mae_shmu = 0
    mse_shmu = 0
    start = 0
    predicted_all = np.array([])
    train_end = window_len * interval

    while (train_end < data_len):
        # Check how many prediction we can make within same ref_date
        ref_date = data.reference_date[train_end]

        pred_length = 0
        while (data.reference_date[train_end + pred_length] == ref_date):
            pred_length += 1
            # Out of bounds
            if (pred_length + train_end >= data_len):
                break

        x_train_sets = []
        y_train_sets = []

        for i in range(pred_length):
            x_train_sets.append(x.iloc[start + i:train_end:interval, :])
            y_train_sets.append(y.iloc[start + i:train_end:interval])

        x_test = x.iloc[train_end:train_end + pred_length, :]
        y_test = y.iloc[train_end:train_end + pred_length]

        for i in range(pred_length):
            weights = None
            if (weight):
                w = list(reversed([math.sqrt(weight ** j)
                                   for j in range(x_train_sets[i].shape[0])]))
                weights = np.array(w)

            if (weights):
                model.fit(x_train_sets[i], y_train_sets[
                          i], sample_weight=weights)
            else:
                model.fit(x_train_sets[i], y_train_sets[i])

            y_predicted = model.predict(
                x_test.iloc[i, :].values.reshape(1, -1))

            # add into predicted_all
            predicted_all = np.hstack((predicted_all, y_predicted))

            mae_shmu += np.sum(abs(y_test.iloc[i] -
                                   x_test.future_temp_shmu.iloc[i]))
            mse_shmu += np.sum((y_test.iloc[i] -
                                x_test.future_temp_shmu.iloc[i]) ** 2)

            mae_predict += np.sum(abs(y_test.iloc[i] - y_predicted))
            mse_predict += np.sum((y_test.iloc[i] - y_predicted) ** 2)

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
    }
