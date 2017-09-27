import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.ensemble as dt
import sklearn.neural_network as nn
import sklearn.neighbors as ng
import pandas as pd
import os

from feature_utils import add_moments
from feature_utils import shmu_error_prediction_time_moment
from feature_utils import feature_lagged_by_hours
from feature_utils import feature_lagged_by_hours_p_time
from feature_utils import shmu_prediction_time_error
from feature_utils import add_shmu_error
from feature_utils import add_min_max

from utils import get_bias, save_predictions, save_bias
from utils import save_errors, predict, predict_test
from parsers import get_predict_parser

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    parser = get_predict_parser()
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
    autocorrect = args.autocorrect
    verbose = args.verbose
    moments = args.moments
    min_max = args.min_max
    use_cache = args.use_cache
    skip_predictions = int(args.skip_predictions)
    shmu_error_moment = args.shmu_error_moment

    # feature switches
    shmu_error = int(args.shmu_error)

    shmu_error_p_time = args.shmu_error_p_time
    splitted = shmu_error_p_time.split(':')
    shmu_error_p_time_lags = int(splitted[0])
    shmu_error_p_time_lag_by = int(splitted[1])
    shmu_error_p_time_exp = float(splitted[2])

    feature_p_time = args.feature_p_time
    feature_p_time_name = None
    feature_p_time_lags = 0
    feature_p_time_lag_by = 0
    if (feature_p_time is not None):
        splitted = feature_p_time.split(':')
        feature_p_time_name = splitted[2]
        feature_p_time_lag_by = int(splitted[1])
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

    data = None
    x = None
    y = None

    if (not use_cache):
        data = pd.read_csv(args.data_file, delimiter=';')
        print('Preparing data')

        data = shmu_prediction_time_error(data,
                                          shmu_error_p_time_lags,
                                          shmu_error_p_time_lag_by,
                                          shmu_error_p_time_exp)

        data = feature_lagged_by_hours_p_time(data,
                                              feature_p_time_name,
                                              feature_p_time_lags,
                                              feature_p_time_lag_by)

        data = feature_lagged_by_hours(data,
                                       feature_name, feature_lags,
                                       feature_lag_by)

        data = add_moments(data, moments)
        data = add_min_max(data, min_max)
        data = add_shmu_error(data, shmu_error)
        data = shmu_error_prediction_time_moment(data, shmu_error_moment)

        '''
        fieldsToDrop = ['p_time_temp', 'p_time_humidity', 'p_time_pressure',
                        'p_time_rainfall_last_hour', 'p_time_wind_speed',
                        'p_time_wind_direction']

        '''
        fieldsToDrop = ['current_temp', 'current_humidity', 'current_pressure',
                        'current_rainfall_last_hour', 'current_wind_speed',
                        'current_wind_direction']

        fieldsToDrop.append('future_temp')
        fieldsToDrop.append('validity_date')
        fieldsToDrop.append('reference_date')

        fieldsToDrop.append('p_time_rainfall_last_hour')

        fieldsToDrop.append('p_time_humidity')

        fieldsToDrop.append('p_time_pressure')

        fieldsToDrop.append('p_time_wind_speed')

        fieldsToDrop.append('p_time_wind_direction')

        y = data.future_temp
        x = data.drop(fieldsToDrop, axis=1)
    else:
        data = pd.read_csv('cached_data.csv', delimiter=';', index_col=0)
        x = pd.read_csv('cached_x.csv', delimiter=';', index_col=0)
        y = pd.read_csv('cached_y.csv', delimiter=';', index_col=0)
        y = pd.Series(y.values.ravel())  # TODO remove this hack

    data.to_csv('cached_data.csv', sep=';')
    y.to_csv('cached_y.csv', sep=';', header='future_temp')
    x.to_csv('cached_x.csv', sep=';')

    # for testing
    # data = data.iloc[12:, :].reset_index(drop=True)

    print('Features used', x.columns, x.shape)

    models = []
    if (model_type == 'svr'):
        models.append(svm.SVR(C=1, kernel='linear', epsilon=0.05))
        '''
        models.append(svm.SVR(C=1, kernel='rbf', epsilon=0.1,
                              gamma=0.05))
        '''
    elif (model_type == 'reg'):
        models.append(lm.LinearRegression(fit_intercept=fit_intercept))
    elif (model_type == 'rf'):
        models.append(dt.RandomForestRegressor(n_estimators=50, max_depth=5))
        '''
        models.append(dt.GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1, max_depth=5))
        '''
    elif (model_type == 'nn'):
        models.append(nn.MLPRegressor(hidden_layer_sizes=(
            20,), max_iter=15, activation='relu',
            solver='lbfgs', alpha=0.001))
        models.append(nn.MLPRegressor(hidden_layer_sizes=(
            20,), max_iter=15, activation='relu',
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
        # models.append(lm.LinearRegression(fit_intercept=fit_intercept))
        models.append(lm.Lasso(alpha=0.1, copy_X=True, fit_intercept=True,
                               max_iter=50, normalize=False))

    elif (model_type == 'ens-ens'):
        models.append(dt.GradientBoostingRegressor(
            n_estimators=20,
            learning_rate=0.1, max_depth=2))
        models.append(dt.RandomForestRegressor(n_estimators=20, max_depth=2))

    stats = predict(data, x, y, weight, models, length, step, diff,
                    norm, average_models, autocorrect, verbose,
                    skip_predictions)

    # stats = predict_test(data, x, y, weight, models, length, step)

    print('BIAS (temperature) in data {0:.2f}'.format(get_bias(
        real=data.future_temp, predicted=data.future_temp_shmu)))

    print('MAE SHMU {0:.2f}'.format(stats['mae_shmu']))
    print('MAE PREDICT {0:.4f}'.format(stats['mae_predict']))
    print('MSE SHMU {0:.2f}'.format(stats['mse_shmu']))
    print('MSE PREDICT {0:.4f}'.format(stats['mse_predict']))
    print('Model bias {0:.4f}'.format(stats['model_bias']))

    predicted = stats['predicted_all']
    predictions_count = stats['predictions_count']
    cum_mse = stats['cum_mse']
    cum_mae = stats['cum_mae']
    cum_bias = stats['cum_bias']

    print('Predictions count', predictions_count)

    predicted_errors = predicted - data.future_temp[-predictions_count:]
    shmu_errors = data.future_temp_shmu[-predictions_count:] - \
        data.future_temp[-predictions_count:]

    save_predictions(real_values=data.future_temp,
                     predicted_values=stats['predicted_all'],
                     shmu_predictions=data.future_temp_shmu)

    save_errors(predicted_errors, shmu_errors, cum_mse, cum_mae)
    save_bias(cum_bias)
