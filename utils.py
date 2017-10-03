import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

from feature_utils import get_autocorrect_func


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
            norm=False, average_models=False, autocorrect=False,
            verbose=False, skip_predictions=0):
    '''
    TODO refactor:
    This function is extremely long and does too much things.

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
    iterations = 0

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
    cum_bias = []

    autocorrect_func = get_autocorrect_func(autocorrect)

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
                autocorrect_col = autocorrect_func(model_errors, i,
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
            x_test['autocorrect'] = pd.Series(autocorrect_func(
                model_errors, is_test_set=True,
                test_set_length=pred_length), index=x_test.index)

        x_test_orig = x_orig.iloc[train_end:train_end + pred_length, :]
        y_test = y_orig.iloc[train_end:train_end + pred_length]

        # for testing purpose if autocorrect data are correct
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

            iterations += 1

            if (iterations <= skip_predictions):
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
            cum_bias.append(model_bias / predictions_made)

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
        'cum_mse': cum_mse[200:],
        'cum_mae': cum_mae[200:],
        'cum_bias': cum_bias[200:],
    }


def predict_one_hour_intervals(data, x, y, weight, models, length, step):
    '''
    Use step between measure set to one hour.
    Currently not used as does not seem very reasonable.

    TODO: incorporate into `predict` function
    '''
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
        'model_bias': 0,
    }
