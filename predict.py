import numpy as np
import math
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from autocorrect_features import get_autocorrect_conf
from stable_weather_detection import get_stable_func
from stable_weather_detection import is_error_diff_enough
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _get_train_data(x, y, i, window_period, interval, diff):
    x_train = x[i - (window_period * interval):i:interval, :]
    y_train = y[i - (window_period * interval):i:interval]

    if (diff):
        x_train = np.diff(x_train, axis=0)
        y_train = y_train[1:]

    return x_train, y_train


def _get_test_data(x, y, i, interval, diff):
    x_test = x[i, :]
    y_test = y[i]

    if (diff):
        # +1 is used to include current item
        x_test = np.squeeze(
            np.diff(x[i - interval: i + 1: interval, :], axis=0))

    return x_test, y_test


def _get_weights(weight, x_train):
    if (weight):
        w = list(reversed([math.sqrt(weight ** j)
                           for j in range(x_train.shape[0])]))
        return np.array(w)
    return None


def _contains_missing_data(x_train, y_train, x_test, y_test):
    missing_data_value = -999

    return (x_train.__contains__(missing_data_value) or
            y_train.__contains__(missing_data_value) or
            x_test.__contains__(missing_data_value) or
            y_test.__contains__(missing_data_value))


def _print_position_info(pos):
    if (pos > 0 and (pos % 100) == 0):
        print('At position {}'.format(pos), end="\r")


def _add_autocorrection(x_train, x_test, autocorrect, window_length,
                        window_period, model_errors, position,
                        autocorrect_only_stable, is_stable):
    autocorrect_conf = get_autocorrect_conf(autocorrect)
    if ((autocorrect_conf is None) or
            (autocorrect_only_stable and not is_stable)):
        return (x_train, x_test, True)

    can_use_autocorrect = autocorrect_conf['can_use_auto']
    autocorrect_func = autocorrect_conf['func']
    merge_func = autocorrect_conf['merge']

    autocorrect_ready = can_use_autocorrect(
        model_errors, window_period, window_length)

    if (autocorrect and autocorrect_ready):
        x_train_auto = autocorrect_func(model_errors, position,
                                        window_period, window_length)

        x_test_auto = autocorrect_func(model_errors, is_test_set=True)
        x_train_new, x_test_new = merge_func(x_train, x_test,
                                             x_train_auto, x_test_auto,
                                             window_length)
        return (x_train_new, x_test_new, autocorrect_ready)
    return (x_train, x_test, autocorrect_ready)


def _fit_model(model, x_train, y_train, weight):
    weights = _get_weights(weight, x_train)
    if (weights is not None):
        model.fit(x_train, y_train, sample_weight=weights)
    else:
        model.fit(x_train, y_train)


def _predictions_to_dataframe(predictions):
    return pd.DataFrame.from_items(
        [('validity_date', predictions[0]),
         ('predicted', predictions[1]),
         ], columns=['validity_date', 'predicted'])


def _scale_data(scale, x_train, x_test):
    if (scale):
        scaler = None
        if (scale == 'min-max'):
            scaler = MinMaxScaler()
        elif (scale == 'standard'):
            scaler = StandardScaler()

        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test.reshape(1, -1))
    return (x_train, x_test)


def predict(data, x, y, model, window_length, window_period,
            weight=None, scale=False, autocorrect=False, stable=False,
            stable_func=None, ignore_diff_errors=False,
            autocorrect_only_stable=False, diff=False, skip=0):
    start = window_length * window_period + skip
    predicted_all = [[], []]  # 0 - validity_date, 1 - predicted value
    model_errors = np.array([])
    stable_func = get_stable_func(stable_func)

    train_mse = 0
    train_mae = 0
    r2 = 0

    for i in range(start, x.shape[0]):
        _print_position_info(i)
        val_date = data.validity_date[i]

        x_train, y_train = _get_train_data(
            x, y, i, window_length, window_period, diff=diff)
        x_test, y_test = _get_test_data(x, y, i, window_period, diff=diff)

        if (_contains_missing_data(
                x_train, y_train, x_test, np.matrix(y_test))):
            model_errors = np.array([])
            continue

        is_stable = stable_func(data, i)

        x_train, x_test, autocorrect_ready = _add_autocorrection(
            x_train, x_test, autocorrect, window_length,
            window_period, model_errors, i, autocorrect_only_stable,
            is_stable)

        x_train, x_test = _scale_data(scale, x_train, x_test)

        _fit_model(model, x_train, y_train, weight)
        y_predicted = model.predict(x_test.reshape(1, -1))

        if (diff):
            y_predicted += y_train[-1]

        if (not ((stable and not is_stable) or
                 (autocorrect and not autocorrect_ready) or
                 (autocorrect and ignore_diff_errors and
                  is_error_diff_enough(model_errors)))):
            r2 += model.score(x_train, y_train)
            train_mse += mean_squared_error(model.predict(x_train), y_train)
            train_mae += mean_absolute_error(model.predict(x_train), y_train)
            predicted_all[0].append(val_date)
            predicted_all[1].append(y_predicted[0])

        model_error = (y_predicted - y_test)[0]
        model_errors = np.append(model_errors, model_error)

    print('TRAIN MAE: {0:.4f}'.format(train_mae / len(predicted_all[0])))
    print('TRAIN MSE: {0:.4f}'.format(train_mse / len(predicted_all[0])))
    print('R2: {0:.4f}'.format(r2 / len(predicted_all[0])))

    return _predictions_to_dataframe(predicted_all)
