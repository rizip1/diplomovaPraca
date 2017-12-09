import re
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from autocorrect_features import get_autocorrect_conf
from stable_weather_detection import get_stable_func
from stable_weather_detection import is_error_diff_enough

from scipy.stats.mstats import normaltest
from scipy.stats import norm


improvements = [[] for i in range(24)]
season_improvements = {
    'spring': [[] for i in range(24)],
    'summer': [[] for i in range(24)],
    'autumn': [[] for i in range(24)],
    'winter': [[] for i in range(24)],
}


class Colors:
    BLUE = '\033[94m'
    ENDC = '\033[0m'


def color_print(text, color=Colors.BLUE):
    print(color + text + Colors.ENDC)


def is_stable_weather(data, pos):
    min_hours = 4
    max_dist = 1

    for i in range(min_hours):
        if (abs(data.loc[pos - 24 - i, 'current_temp'] - data.loc[pos - i - 48, 'current_temp']) > max_dist):
            return False
    return True


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


def save_hour_value(all_results, seasonal_results, value, period, month):
    p = period - 1
    all_results[p].append(value)
    if (month in [1, 2, 3]):
        seasonal_results['winter'][p].append(value)
    elif (month in [4, 5, 6]):
        seasonal_results['spring'][p].append(value)
    elif (month in [7, 8, 9]):
        seasonal_results['summer'][p].append(value)
    elif (month in [10, 11, 12]):
        seasonal_results['autumn'][p].append(value)


def plot_hour_results(all_results, seasonal_results, title, file_name):
    morning = []
    afternoon = []
    x = [(i + 1) for i in range(24)]
    for i, values in enumerate(all_results):
        mean = np.mean(values)
        if (i < 12):
            morning.append(mean)
        else:
            afternoon.append(mean)

    plt.figure(figsize=(12, 6))
    plt.plot(x[0:12], morning, 'r')
    plt.plot(x[12:], afternoon, 'g')
    plt.title(title)
    plt.ylabel('Errors')
    plt.xlabel('Hours')
    plt.xticks(x)
    plt.grid()
    plt.savefig('improvement/{}.png'.format(file_name))
    plt.close()

    plt.figure(figsize=(12, 6))

    colors = ['r', 'g', 'b', 'y']
    for index, period in enumerate(['spring', 'summer', 'autumn', 'winter']):
        morning = []
        afternoon = []
        x = [(i + 1) for i in range(24)]
        for i, values in enumerate(seasonal_results[period]):
            mean = np.mean(values)
            if (i < 12):
                morning.append(mean)
            else:
                afternoon.append(mean)

        plt.plot(x[0:12], morning, colors[index], label=period)
        plt.plot(x[12:], afternoon, colors[index])
    plt.title(title)
    plt.ylabel('Errors')
    plt.xlabel('Hours')
    plt.xticks(x)
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1.015), loc=2)
    plt.savefig('improvement/{}_periods.png'.format(file_name))
    plt.close()


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


def plot_normality(errors):
    # based on D’Agostino and Pearson’s test that combines
    # skew and kurtosis to produce an omnibus test of normality.
    print('\nErrors normality p-value: ', normaltest(errors).pvalue)

    mu, std = norm.fit(errors)
    plt.figure(figsize=(12, 6))
    plt.hist(errors, bins=30, normed=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.savefig('other/errors_hist.png')
    plt.close()

    qqplot(errors)
    plt.savefig('other/errors_qq.png')
    plt.close()


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


def can_use_autocorrect(model_errors, interval, window_len):
    period = 24 * 2
    return len(model_errors) > (interval * window_len) + period


def get_best_model(models, x_train, y_train, weights):
    best_model = None
    best_score = -float('inf')
    for m in models:
        score = -float('inf')
        if (weights is not None):
            try:
                m.fit(x_train, y_train, sample_weight=weights)
                score = m.score(x_train, y_train, sample_weight=weights)
            except:
                m.fit(x_train, y_train)
                score = m.score(x_train, y_train)
        else:
            m.fit(x_train, y_train)
            score = m.score(x_train, y_train)
        if (score > best_score):
            best_score = score
            best_model = m
    return best_model


def get_avg_prediction(models, x_train, y_train, x_test, weights):
    predicted = 0
    for m in models:
        try:
            m.fit(x_train, y_train, sample_weight=weights)
        except:
            m.fit(x_train, y_train)
        predicted += m.predict(x_test)
    return predicted / len(models)


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


def save_errors(predicted_errors, shmu_errors, cum_mse, cum_mae,
                model_hour_errors):
    max_count = 500

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot(predicted_errors, 'k', label='predicted errors')
    plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
    plt.title('Temperature errors')
    plt.ylabel('Error')
    plt.xlabel('Samples')
    plt.savefig('other/errors/errors.png')
    plt.close()

    print('\nSaving error plots ...')
    for i in range(0, len(predicted_errors), max_count):
        end = min(i + max_count, len(predicted_errors))
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.plot(predicted_errors[i:end], 'k', label='predicted errors')
        plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
        plt.title('Temperature errors')
        plt.ylabel('Error')
        plt.xlabel('Samples')
        plt.savefig('other/errors/errors_{}.png'.format(i))
        plt.close()

    print('\n Durbin watson stats:')
    for i in range(24):
        print(durbin_watson(model_hour_errors[i]))
        plot_acf(model_hour_errors[i], lags=60, alpha=0.01)
        plt.savefig('other/errors/errors_autocorr_{}.png'.format(i))
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
    plt.savefig('other/errors/cum_errors.png')
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


def predict_new(data, x, y, model, window_length, window_period,
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


def predict(data, x, y, weight, models, shmu_predictions, window_len,
            interval=12, diff=False,
            norm=False, average_models=True, autocorrect=False,
            verbose=False, skip_predictions=0):
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

    data_len = x.shape[0]
    predictions_made = 0
    iterations = 0
    mae_predict = 0
    mse_predict = 0
    mae_shmu = 0
    mse_shmu = 0
    predicted_all = np.array([])
    start = window_len * interval
    model_errors = np.array([])
    model_predictions = np.array([])
    model_bias = 0
    cum_mse = []
    cum_mae = []
    cum_bias = []
    shmu_bias = 0
    autocorrect_func = get_autocorrect_func(autocorrect)
    model_hour_errors = [[] for i in range(24)]

    for i in range(start, data_len):
        if (verbose and predictions_made > 0 and (i % 100) == 0):
            print('current pos', i, mae_predict /
                  predictions_made, mse_predict / predictions_made,
                  model_bias / predictions_made)

        val_date = data.validity_date[i]
        val_date_hour = parse_hour(val_date)
        val_date_month = parse_month(val_date)
        std, mean = [None, None]

        autocorrect_ready = can_use_autocorrect(
            model_errors, interval, window_len)

        x_train, y_train = get_train_data(
            x, y, i, start, interval, diff=diff)

        if (autocorrect and autocorrect_ready):
            autocorrect_col = autocorrect_func(model_errors, i,
                                               interval, window_len)
            x_train['autocorrect'] = pd.Series(
                autocorrect_col, index=x_train.index)
        if (norm):
            std = x_train.std()
            mean = x_train.mean()
            x_train = (x_train - mean) / std

        x_test, y_test = get_test_data(x, y, i, interval, diff=diff)

        if (autocorrect and autocorrect_ready):
            x_test['autocorrect'] = autocorrect_func(
                model_errors, is_test_set=True)

        # uncomment for testing purpose if autocorrect data are correct
        '''
        if (predictions_made > 0):
            save_autocorrect_state(model_errors, x_train, x_test)
        '''

        if (contains_missing_data(
                x_train, y_train,
                x_test, np.matrix(y_test))):
            continue

        best_model = None
        weights = get_weights(weight, x_train)
        y_predicted = 0

        if (norm):
            x_test = (x_test - mean) / std

        if (not average_models):
            best_model = get_best_model(models, x_train, y_train, weights)
            y_predicted = best_model.predict(x_test.reshape(1, -1))
        else:
            y_predicted = get_avg_prediction(models, x_train, y_train,
                                             x_test.reshape(1, -1), weights)

        model_error = (y_predicted - y_test)[0]
        model_errors = np.append(model_errors, model_error)
        model_predictions = np.append(model_predictions, y_predicted)
        model_hour_errors[val_date_hour - 1].append(model_error)

        if (autocorrect and not autocorrect_ready):
            continue

        iterations += 1
        if (iterations <= skip_predictions):
            continue

        # add into predicted_all
        predicted_all = np.hstack((predicted_all, y_predicted))
        predictions_made += 1

        shmu_value = shmu_predictions[i]
        shmu_error = abs(y_test - shmu_value)
        predicted_error = abs(y_test - y_predicted)
        improvement = (shmu_error - predicted_error)[0]
        save_hour_value(improvements, season_improvements,
                        improvement, val_date_hour, val_date_month)

        mae_shmu += np.sum(abs(y_test - shmu_value))
        mse_shmu += np.sum((y_test - shmu_value) ** 2)

        mae_predict += np.sum(abs(y_test - y_predicted))
        mse_predict += np.sum((y_test - y_predicted) ** 2)
        model_bias += (y_predicted - y_test)[0]
        shmu_bias += (shmu_value - y_test)

        cum_mse.append(mse_predict / predictions_made)
        cum_mae.append(mae_predict / predictions_made)
        cum_bias.append(model_bias / predictions_made)

    save_improvement_to_file()
    plot_hour_results(improvements, season_improvements,
                      'Improvements', 'improvements')
    plot_normality(model_errors)
    pd.Series(model_errors).to_csv(
        'other/compare_errors/errors.csv', index=False)

    return {
        'model_hour_errors': model_hour_errors,
        'mae_predict': mae_predict / predictions_made,
        'mae_shmu': mae_shmu / predictions_made,
        'mse_predict': mse_predict / predictions_made,
        'mse_shmu': mse_shmu / predictions_made,
        'predicted_all': np.array(predicted_all),
        'predictions_count': predictions_made,
        'model_bias': model_bias / predictions_made,
        'shmu_bias': shmu_bias / predictions_made,
        'cum_mse': cum_mse[200:],  # removing first values improves readability
        'cum_mae': cum_mae[200:],
        'cum_bias': cum_bias[200:],
    }
