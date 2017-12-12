import re
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from autocorrect_features import get_autocorrect_conf
from stable_weather_detection import get_stable_func
from stable_weather_detection import is_error_diff_enough


class Colors:
    BLUE = '\033[94m'
    ENDC = '\033[0m'


def color_print(text, color=Colors.BLUE):
    print(color + text + Colors.ENDC)


def get_train_data(x, y, i, start, interval, diff=False):
    x_train = x[i - start:i:interval, :]
    y_train = y[i - start:i:interval]

    if (diff):
        x_train = np.diff(x_train, axis=0)
        y_train = y_train[1:]

    return x_train, y_train


def get_test_data(x, y, i, interval, diff=False):
    x_test = x[i, :]
    y_test = y[i]

    if (diff):
        # +1 is used to include current item
        x_test = np.squeeze(
            np.diff(x[i - interval: i + 1: interval, :], axis=0))

    return x_test, y_test


'''
def save_improvement_to_file():
    total_improvements = 0
    total_draws = 0
    total_worse = 0
    morning = []
    afternoon = []

    for index, values in enumerate(improvements):
        for v in values:
            if (v < 0):
                total_worse += 1
            elif (v > 0):
                total_improvements += 1
            else:
                total_draws += 1

        val = np.mean(values)
        if (index < 12):
            morning.append(val)
        else:
            afternoon.append(val)

    with open('improvement/total_improvemets.txt', 'w') as f:
        f.write('Morning\n')
        for v in morning:
            f.write('{},\n'.format(v))

        f.write('Afternoon\n')
        for v in afternoon:
            f.write('{},\n'.format(v))

        f.write('\nTotal improvements: {}\n'.format(total_improvements))
        f.write('Total worse: {}\n'.format(total_worse))
        f.write('Total draws: {}\n'.format(total_draws))
        f.write('Total records: {}\n'.format(
                total_draws + total_improvements + total_worse))

    for season, hour_values in season_improvements.items():
        total_improvements = 0
        total_draws = 0
        total_worse = 0

        morning = []
        afternoon = []
        for index, values in enumerate(hour_values):
            for v in values:
                if (v < 0):
                    total_worse += 1
                elif (v > 0):
                    total_improvements += 1
                else:
                    total_draws += 1

            val = np.mean(values)
            if (index < 12):
                morning.append(val)
            else:
                afternoon.append(val)
        with open('improvement/{}_improvemets.txt'.format(season), 'w') as f:
            f.write('{}\n'.format(season))

            f.write('Morning\n')
            for v in morning:
                f.write('{},\n'.format(v))

            f.write('Afternoon\n')
            for v in afternoon:
                f.write('{},\n'.format(v))

            f.write('\nTotal improvements: {}\n'.format(total_improvements))
            f.write('Total worse: {}\n'.format(total_worse))
            f.write('Total draws: {}\n'.format(total_draws))
            f.write('Total records: {}\n'.format(
                total_draws + total_improvements + total_worse))

            frame = pd.DataFrame()
            frame['morning'] = pd.Series(morning)
            frame['afternoon'] = pd.Series(afternoon)
            frame.to_csv(
                'improvement/{}_improvemets.csv'.format(season), index=False)
'''


def save_autocorrect_state(model_errors, x_train, x_test):
    '''
    Used for testing purpose to check if autocorrection
    errors are picked the right way.
    Will finish execution of program when finished.
    '''
    pd.Series(model_errors).to_csv('test/model_errors.csv')
    x_train.to_csv('test/train.csv')
    x_test.to_csv('test/test.csv')
    import os
    os.sys.exit(1)


def parse_hour(date):
    m = re.search(
        r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
        date)
    hour = int(m.group(1))
    if (hour == 0):
        hour = 24
    return hour


def parse_month(date):
    m = re.search(
        r'^[0-9]{4}-([0-9]{2})-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$',
        date)
    return int(m.group(1))


def get_weights(weight, x_train):
    if (weight):
        w = list(reversed([math.sqrt(weight ** j)
                           for j in range(x_train.shape[0])]))
        return np.array(w)
    return None


def contains_missing_data(x_train, y_train, x_test, y_test):
    missing_data_value = -999

    return (x_train.__contains__(missing_data_value) or
            y_train.__contains__(missing_data_value) or
            x_test.__contains__(missing_data_value) or
            y_test.__contains__(missing_data_value))


def save_predictions(real_values, predicted_values, shmu_predictions):
    real_values_x = [i for i in range(
        len(real_values) - (len(real_values) - len(predicted_values)))]

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot(predicted_values, 'or', label='Predicted values (Our model)')
    plt.plot(real_values_x, real_values[-len(predicted_values):],
             'ok', label='Real values')
    plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
    plt.title('Temperature predictions')
    plt.ylabel('Temperature')
    plt.xlabel('Samples')
    plt.savefig('other/predictions.png')
    plt.close()


def save_bias(cum_bias):
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot(cum_bias, 'r', label='Cummulative bias')
    plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
    plt.title('Cummulative bias')
    plt.ylabel('Bias')
    plt.xlabel('Prediction id')
    plt.savefig('other/cum_bias.png')
    plt.close()


def print_position_info(pos):
    if (pos > 0 and (pos % 100) == 0):
        print('At position {}'.format(pos), end="\r")


def add_autocorrection(x_train, x_test, autocorrect, window_length,
                       window_period, model_errors, position,
                       autocorrect_only_stable, is_stable):
    autocorrect_conf = get_autocorrect_conf(autocorrect)
    if ((autocorrect_conf is None) or
            (autocorrect_only_stable and not is_stable)):
        return (x_train, x_test, False)

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


def fit_model(model, x_train, y_train, weight):
    weights = get_weights(weight, x_train)
    if (weights is not None):
        model.fit(x_train, y_train, sample_weight=weights)
    else:
        model.fit(x_train, y_train)


def predictions_to_dataframe(predictions):
    return pd.DataFrame.from_items(
        [('validity_date', predictions[0]),
         ('predicted', predictions[1]),
         ], columns=['validity_date', 'predicted'])


def scale(scaler, x_train, x_test):
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test.reshape(1, -1))
    return (x_train, x_test)


def predict(data, x, y, model, window_length, window_period,
            weight=None, autocorrect=False, stable=False,
            stable_func=None, ignore_diff_errors=False,
            autocorrect_only_stable=False):
    start = window_length * window_period
    predicted_all = [[], []]  # 0 - validity_date, 1 - predicted value
    model_errors = np.array([])
    stable_func = get_stable_func(stable_func)

    for i in range(start, x.shape[0]):
        print_position_info(i)
        val_date = data.validity_date[i]

        x_train, y_train = get_train_data(
            x, y, i, start, window_period, diff=False)
        x_test, y_test = get_test_data(x, y, i, window_period, diff=False)

        if (contains_missing_data(
                x_train, y_train, x_test, np.matrix(y_test))):
            model_errors = np.array([])
            continue

        is_stable = stable_func(data, i)

        x_train, x_test, autocorrect_ready = add_autocorrection(
            x_train, x_test, autocorrect, window_length,
            window_period, model_errors, i, autocorrect_only_stable,
            is_stable)

        # x_train, x_test = scale(StandardScaler(), x_train, x_test)

        fit_model(model, x_train, y_train, weight)
        y_predicted = model.predict(x_test.reshape(1, -1))

        if (not ((stable and not is_stable) or
                 (autocorrect and not autocorrect_ready) or
                 (autocorrect and ignore_diff_errors and
                  is_error_diff_enough(model_errors)))):
            predicted_all[0].append(val_date)
            predicted_all[1].append(y_predicted[0])

        model_error = (y_predicted - y_test)[0]
        model_errors = np.append(model_errors, model_error)

    return predictions_to_dataframe(predicted_all)
