import re
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

from feature_utils import get_autocorrect_func

improvements = [[] for i in range(24)]
season_improvements = {
    'spring': [[] for i in range(24)],
    'summer': [[] for i in range(24)],
    'autumn': [[] for i in range(24)],
    'winter': [[] for i in range(24)],
}


def save_hour_value(all_results, seasonal_results, value, period, month):
    all_results[period].append(value)
    if (month in [1, 2, 3]):
        seasonal_results['winter'][period].append(value)
    elif (month in [4, 5, 6]):
        seasonal_results['spring'][period].append(value)
    elif (month in [7, 8, 9]):
        seasonal_results['summer'][period].append(value)
    elif (month in [10, 11, 12]):
        seasonal_results['autumn'][period].append(value)


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
    return int(m.group(1))


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
    period = 24
    error_window = 12
    return len(model_errors) > (interval * window_len) + (2 * period) + \
        error_window


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


def save_errors(predicted_errors, shmu_errors, cum_mse, cum_mae):
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot(predicted_errors, 'k', label='predicted errors')
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


def predict(data, x, y, weight, models, window_len, interval=12, diff=False,
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

    x_orig, y_orig, x_diff, y_diff = [None for i in range(4)]
    if (diff):
        x_orig = x.iloc[interval:]
        y_orig = y.iloc[interval:]
        x_diff = x.diff(periods=interval).iloc[interval:]
        y_diff = y.diff(periods=interval).iloc[interval:]
    else:
        x_orig, y_orig, x_diff, y_diff = [
            x.copy(), y.copy(), x.copy(), y.copy()]

    data_len = x_diff.shape[0]
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
    autocorrect_func = get_autocorrect_func(autocorrect)

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

        x_train = x_diff.iloc[i - start:i:interval, :]
        y_train_orig = y_orig.iloc[i - start:i:interval]
        y_train = y_diff.iloc[i - start:i:interval]

        if (autocorrect and autocorrect_ready):
            autocorrect_col = autocorrect_func(model_errors, i,
                                               interval, window_len)
            x_train['autocorrect'] = pd.Series(
                autocorrect_col, index=x_train.index)
        if (norm):
            std = x_train.std()
            mean = x_train.mean()
            x_train = (x_train - mean) / std

        x_test = x_diff.iloc[i, :]
        x_test_orig = x_orig.iloc[i, :]
        y_test = y_orig.iloc[i]

        if (autocorrect and autocorrect_ready):
            x_test['autocorrect'] = autocorrect_func(
                model_errors, is_test_set=True)

        # uncomment for testing purpose if autocorrect data are correct
        '''
        if (predictions_made > 0):
            save_autocorrect_state(model_errors, x_train, x_test)
        '''

        if (contains_missing_data(
                x_train.values, y_train.values,
                x_test.values, np.matrix(y_test))):
            continue

        best_model = None
        weights = get_weights(weight, x_train)
        y_predicted = 0

        if (norm):
            x_test = (x_test - mean) / std

        if (not average_models):
            best_model = get_best_model(models, x_train, y_train, weights)
            y_predicted = best_model.predict(x_test.values.reshape(1, -1))
        else:
            y_predicted = get_avg_prediction(models, x_train, y_train,
                                             x_test.values.reshape(1, -1))

        if (diff):
            y_predicted += y_train_orig.iloc[-1]

        model_errors = np.append(model_errors, (y_predicted - y_test)[0])
        model_predictions = np.append(model_predictions, y_predicted)

        if (autocorrect and not autocorrect_ready):
            continue

        iterations += 1
        if (iterations <= skip_predictions):
            continue

        # add into predicted_all
        predicted_all = np.hstack((predicted_all, y_predicted))
        predictions_made += 1

        shmu_value = x_test_orig.future_temp_shmu
        shmu_error = abs(y_test - shmu_value)
        predicted_error = abs(y_test - y_predicted)
        improvement = (shmu_error - predicted_error)[0]
        save_hour_value(improvements, season_improvements,
                        improvement, val_date_hour, val_date_month)

        mae_shmu += np.sum(abs(y_test - shmu_value))
        mse_shmu += np.sum((y_test - shmu_value) ** 2)

        mae_predict += np.sum(abs(y_test - y_predicted))
        mse_predict += np.sum((y_test - y_predicted) ** 2)
        model_bias += (y_test - y_predicted)[0]

        cum_mse.append(mse_predict / predictions_made)
        cum_mae.append(mae_predict / predictions_made)
        cum_bias.append(model_bias / predictions_made)

    save_improvement_to_file()
    plot_hour_results(improvements, season_improvements,
                      'Improvements', 'improvements')

    return {
        'mae_predict': mae_predict / predictions_made,
        'mae_shmu': mae_shmu / predictions_made,
        'mse_predict': mse_predict / predictions_made,
        'mse_shmu': mse_shmu / predictions_made,
        'predicted_all': np.array(predicted_all),
        'predictions_count': predictions_made,
        'model_bias': model_bias / predictions_made,
        'cum_mse': cum_mse[200:],  # removing first values improves readability
        'cum_mae': cum_mae[200:],
        'cum_bias': cum_bias[200:],
    }
