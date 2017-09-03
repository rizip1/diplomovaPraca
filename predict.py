import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.ensemble as dt
import sklearn.neural_network as nn
import sklearn.neighbors as ng
import pandas as pd
import os

from feature_utils import feature_lagged_by_hours
from feature_utils import feature_lagged_by_hours_p_time
from feature_utils import shmu_prediction_time_error

from utils import get_bias, save_predictions, get_parser
from utils import save_errors, predict

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists('./other'):
        os.makedirs('./other')

    # load command line arguments
    weight = (args.weight_coef, None)[args.weight_coef is None]
    fit_intercept = not args.no_intercept
    mode = args.mode
    length = int(args.length)
    model_type = args.model
    lags = int(args.lags)
    step = int(args.step)
    diff = args.diff
    norm = args.norm
    average_models = args.average_models
    autoreg = args.autoreg
    verbose = args.verbose

    # feature switches
    shmu_error_p_time = args.shmu_error_p_time
    splitted = shmu_error_p_time.split(':')
    shmu_error_p_time_lags = int(splitted[0])
    shmu_error_p_time_lag_by = int(splitted[1])

    feature_p_time = args.feature_p_time
    feature_p_time_name = None
    feature_p_time_lags = 0
    if (feature_p_time is not None):
        splitted = feature_p_time.split(':')
        feature_p_time_name = splitted[1]
        feature_p_time_lags = int(splitted[0])

    feature = args.feature
    feature_name = None
    feature_lags = 0
    feature_lag_by = 0
    if (feature is not None):
        splitted = feature.split(':')
        feature_name = splitted[2]
        feature_lags = int(splitted[0])
        feature_lag_by = int(splitted[1])

    data = pd.read_csv(args.data_file, delimiter=';')

    print('Preparing data')

    data = shmu_prediction_time_error(data,
                                      shmu_error_p_time_lags,
                                      shmu_error_p_time_lag_by)

    data = feature_lagged_by_hours_p_time(data,
                                          feature_p_time_name,
                                          feature_p_time_lags)

    data = feature_lagged_by_hours(data,
                                   feature_name, feature_lags,
                                   feature_lag_by)
    # data = add_func(data)

    # data = data.iloc[15000:, :].reset_index(drop=True)

    y = data.future_temp

    fieldsToDrop = ['future_temp', 'validity_date', 'reference_date']
    # fieldsToDrop = ['future_temp', 'reference_date']

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

    models = []
    if (model_type == 'svr'):
        models.append(svm.SVR(C=1, kernel='linear', epsilon=0.1))
    elif (model_type == 'reg'):
        models.append(lm.LinearRegression(fit_intercept=fit_intercept))
    elif (model_type == 'rf'):
        models.append(dt.RandomForestRegressor(n_estimators=20, max_depth=3))
    elif (model_type == 'nn'):
        models.append(nn.MLPRegressor(hidden_layer_sizes=(
            10,), max_iter=30, activation='logistic',
            solver='lbfgs', alpha=0.001))

        models.append(nn.MLPRegressor(hidden_layer_sizes=(
            5,), max_iter=30, activation='relu',
            solver='lbfgs', alpha=0.001))

        models.append(nn.MLPRegressor(hidden_layer_sizes=(
            20,), max_iter=15, activation='relu',
            solver='lbfgs', alpha=0.001))
    elif (model_type == 'kn'):
        models.append(ng.KNeighborsRegressor())
    elif (model_type == 'ens'):
        models.append(lm.LinearRegression(fit_intercept=fit_intercept))
        models.append(svm.SVR(C=1, kernel='rbf', epsilon=0.1,
                              gamma=0.05))
        models.append(nn.MLPRegressor(hidden_layer_sizes=(
            20,), max_iter=15, activation='relu',
            solver='lbfgs', alpha=0.001))
        models.append(ng.KNeighborsRegressor())
        models.append(lm.Lasso(alpha=0.5, copy_X=True, fit_intercept=True,
                               max_iter=10, normalize=False))
        models.append(dt.GradientBoostingRegressor(
            n_estimators=20,
            learning_rate=0.1, max_depth=2))
        models.append(dt.RandomForestRegressor(n_estimators=20, max_depth=2))
    elif (model_type == 'ens-linear'):
        models.append(lm.LinearRegression(fit_intercept=fit_intercept))
        models.append(lm.Lasso(alpha=0.5, copy_X=True, fit_intercept=True,
                               max_iter=10, normalize=False))

    elif (model_type == 'ens-ens'):
        models.append(dt.GradientBoostingRegressor(
            n_estimators=20,
            learning_rate=0.1, max_depth=2))
        models.append(dt.RandomForestRegressor(n_estimators=20, max_depth=2))

    stats = predict(data, x, y, weight, models, length, step, diff,
                    norm, average_models, autoreg, verbose)

    print('BIAS (temperature) in data {0:.2f}'.format(get_bias(
        real=data.future_temp, predicted=data.future_temp_shmu)))

    print('MAE SHMU {0:.2f}'.format(stats['mae_shmu']))
    print('MAE PREDICT {0:.4f}'.format(stats['mae_predict']))
    print('MSE SHMU {0:.2f}'.format(stats['mse_shmu']))
    print('MSE PREDICT {0:.4f}'.format(stats['mse_predict']))
    print('Model bias {0:.4f}'.format(stats['model_bias']))

    predicted = stats['predicted_all']
    predictions_count = stats['predictions_count']
    predicted_errors = predicted - data.future_temp[-predictions_count:]
    shmu_errors = data.future_temp_shmu[-predictions_count:] - \
        data.future_temp[-predictions_count:]

    save_predictions(real_values=data.future_temp,
                     predicted_values=stats['predicted_all'],
                     shmu_predictions=data.future_temp_shmu)

    save_errors(predicted_errors, shmu_errors)
