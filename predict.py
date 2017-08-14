import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.ensemble as dt
import pandas as pd
import os
import re
import math

from utils import get_bias, save_predictions, get_parser
from utils import save_errors, predict_12_window


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def add_temp_day_lags(data, temp_day_lag):
    data_size = data.shape[0]
    new_start = temp_day_lag * 24
    for i in range(temp_day_lag):
        # add column of zeros for each lag
        new_col = 'temp_day_lag_{}'.format(i + 1)
        data[new_col] = 0  # will set all rows to zero

        start = 24 * (i + 1)
        for j in range(start, data.shape[0]):
            data.loc[j, new_col] = data.loc[j - start, 'future_temp']

    # get rid of values that dont have lagged values
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def add_temp_hour_lags(data, temp_hour_lag):
    data_size = data.shape[0]
    new_start = 12 + temp_hour_lag

    for i in range(temp_hour_lag):
        # add column of zeros for each lag
        new_col = 'temp_hour_lag_{}'.format(i + 1)
        data[new_col] = 0  # will set all rows to zero

        start = i + 12 + 1
        for j in range(start, data.shape[0]):
            ref_date = data.loc[j, 'validity_date']
            m = re.search(
                r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
                ref_date)
            hour = int(m.group(1))
            if (hour > 12):
                hour -= 12

            data.loc[j, new_col] = data.loc[j - hour - (i + 1), 'future_temp']

    # get rid of values that dont have lagged values
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists('./other'):
        os.makedirs('./other')

    # load command line arguments
    weight = (args.weight_coef, None)[args.weight_coef is None]
    fit_intercept = (True, False)[args.intercept is None]
    mode = args.mode
    length = int(args.length)
    model_type = args.model
    lags = int(args.lags)
    temp_day_lag = int(args.temp_day_lag)
    temp_hour_lag = int(args.temp_hour_lag)

    data = pd.read_csv(args.data_file, delimiter=';')

    data = add_temp_day_lags(data, temp_day_lag)
    data = add_temp_hour_lags(data, temp_hour_lag)

    y = data.future_temp

    fieldsToDrop = ['future_temp', 'validity_date', 'reference_date']

    # usually there are lot of missing data
    fieldsToDrop.append('rainfall_last_hour')

    fieldsToDrop.append('pressure')

    # cause strange errors
    fieldsToDrop.append('humidity')

    # cause strange errors
    fieldsToDrop.append('wind_speed')

    # cause strange errors
    fieldsToDrop.append('wind_direction')

    x = data.drop(fieldsToDrop, axis=1)

    print('Features used', x.columns)

    model = None
    if (model_type == 'svr'):
        model = svm.SVR(C=300, kernel='linear')
    elif (model_type == 'reg'):
        model = lm.LinearRegression(fit_intercept=fit_intercept)
    elif (model_type == 'rf'):
        model = dt.RandomForestRegressor(n_estimators=100, max_depth=5)

    stats = predict_12_window(data, x, y, weight, model, length, lags)

    print('BIAS (temperature) in data {0:.2f}'.format(get_bias(
        real=data.future_temp, predicted=data.future_temp_shmu)))

    print('MAE SHMU {0:.2f}'.format(stats['mae_shmu']))
    print('MAE PREDICT {0:.2f}'.format(stats['mae_predict']))
    print('MSE SHMU {0:.2f}'.format(stats['mse_shmu']))
    print('MSE PREDICT {0:.2f}'.format(stats['mse_predict']))

    predicted = stats['predicted_all']
    predictions_count = stats['predictions_count']
    predicted_errors = predicted - data.future_temp[-predictions_count:]
    shmu_errors = data.future_temp_shmu[-predictions_count:] - \
        data.future_temp[-predictions_count:]

    save_predictions(real_values=data.future_temp,
                     predicted_values=stats['predicted_all'],
                     shmu_predictions=data.future_temp_shmu)

    save_errors(predicted_errors, shmu_errors)
