import os
import pandas as pd

import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.ensemble as dt
import sklearn.neural_network as nn
import sklearn.neighbors as ng

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

from feature_utils import add_moments
from feature_utils import add_min_max
from feature_utils import shmu_error_prediction_time_moment
from feature_utils import feature_lagged_by_hours
from feature_utils import feature_lagged_by_hours_p_time
from feature_utils import shmu_prediction_time_error
from feature_utils import add_shmu_error
from feature_utils import add_morning_and_afternoon_temp

from utils import color_print
from predict import predict
from improvements import save_improvements
from conf import config
from error_analysis import save_errors

from constants import PREDICTION_PATH, OTHER_PATH, ERRORS_PATH
from constants import ERRORS_AUTOCOR_PATH, ERRORS_ERRORS_PATH
from constants import IMPROVEMENT_PATH, COMPARED_IMPROVEMENTS_PATH

# The initial pandas warning is statsmodel issue

# Basic model
fieldsToDrop = [
    'current_temp', 'current_humidity', 'current_pressure',
    'current_rainfall_last_hour', 'current_wind_speed',
    'current_wind_direction', 'future_temp', 'validity_date',
    'reference_date', 'p_time_rainfall_last_hour', 'p_time_humidity',
    'p_time_wind_speed', 'p_time_wind_direction', 'p_time_pressure']

'''
fieldsToDrop = [
    'current_temp', 'current_humidity', 'current_pressure',
    'current_rainfall_last_hour', 'current_wind_speed',
    'current_wind_direction', 'future_temp', 'validity_date',
    'reference_date', 'p_time_rainfall_last_hour',
    'p_time_wind_speed', 'p_time_pressure']
'''


def get_model(name, params, x):
    model = None
    if (name == 'ols'):
        model = lm.LinearRegression(fit_intercept=params['fit_intercept'])
    elif (name == 'lasso'):
        # higher alpha = more regularization
        model = lm.Lasso(alpha=0.1, copy_X=True, fit_intercept=True,
                         normalize=False)
    elif (name == 'lasso-cv'):
        cv = TimeSeriesSplit(n_splits=5)
        model = lm.LassoCV(fit_intercept=True, normalize=False, cv=cv)
    elif (name == 'poly-lasso'):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        x = poly.fit_transform(x)
        model = lm.Lasso(alpha=0.01, copy_X=True, fit_intercept=True,
                         normalize=False)
    elif (name == 'ridge'):
        # higher alpha = more regularization
        model = lm.Ridge(alpha=1, copy_X=True, fit_intercept=True,
                         normalize=False)
    elif (name == 'ridge-cv'):
        cv = TimeSeriesSplit(n_splits=5)
        model = lm.RidgeCV(fit_intercept=True, normalize=False, cv=cv)
    elif (name == 'elastic-cv'):
        cv = TimeSeriesSplit(n_splits=5)
        model = lm.ElasticNetCV(cv=cv, fit_intercept=True, normalize=False)
    elif (name == 'bayes-ridge'):
        model = lm.BayesianRidge()
    elif (name == 'svr'):
        # larger C = penalize the cost of missclasification more
        cv = TimeSeriesSplit(n_splits=5)
        parameters = {
            'C': [1, 3, 5, 10, 20, 50],
        }
        s = svm.SVR(kernel='rbf')
        model = GridSearchCV(s, parameters, cv=cv)
        # model = svm.SVR(kernel='linear', C=10, epsilon=0.01)
        # model = svm.SVR(kernel='linear', C=100)
    elif (name == 'knn'):
        model = ng.KNeighborsRegressor(n_neighbors=1)
    elif (name == 'gradient-boost'):
        model = dt.GradientBoostingRegressor(
            n_estimators=30, learning_rate=0.05, max_depth=5)
    elif (name == 'rf'):
        model = dt.RandomForestRegressor(n_estimators=300, max_depth=5)
    elif (name == 'nn'):
        # smaller alpha = more regularization
        cv = TimeSeriesSplit(n_splits=4)
        parameters = {
            'hidden_layer_sizes': [
                [30], [50], [100], [30, 30]
            ],
            'alpha': [1, 0.1, 0.01, 0.001, 0.0001]
        }
        model = nn.MLPRegressor(hidden_layer_sizes=10,
                                activation='relu', solver='lbfgs')
        # model = RandomizedSearchCV(n, parameters, n_iter=3, cv=cv)
    return model, x


def merge_predictions(predictions_all, predicted_values):
    if (predictions_all is not None):
        return pd.merge(predictions_all, predicted_values,
                        on='validity_date', how='outer')
    return predicted_values


def join_date_and_values(predicted_values, validity_date):
    df = pd.DataFrame()
    df['validity_date'] = validity_date
    df['predicted'] = predicted_values
    return df


def save_predictions(result):
    result.to_csv('{}/predictions.csv'.format(PREDICTION_PATH),
                  index=False, sep=';')


def merge_with_measured_and_shmu_predictions(data, final_predictions):
    cols_to_pick = ['validity_date', 'future_temp', 'future_temp_shmu']
    merged = pd.merge(final_predictions, data.loc[:, cols_to_pick],
                      on='validity_date', how='inner').dropna()
    return merged


def show_metrics(result):
    mae_model = mean_absolute_error(result.future_temp, result.predicted)
    mse_model = mean_squared_error(result.future_temp, result.predicted)

    mae_shmu = mean_absolute_error(
        result.future_temp, result.future_temp_shmu)
    mse_shmu = mean_squared_error(
        result.future_temp, result.future_temp_shmu)

    color_print('Model')
    print('MAE: {0:.4f}'.format(mae_model))
    print('MSE: {0:.4f}'.format(mse_model))

    color_print('SHMU')
    print('MAE: {0:.4f}'.format(mae_shmu))
    print('MSE: {0:.4f}'.format(mse_shmu))

    print('\nNumber of predictions', result.shape[0])


def setup_env():
    pd.set_option('display.max_columns', None)

    if not os.path.exists(OTHER_PATH):
        os.makedirs(OTHER_PATH)

    if not os.path.exists(PREDICTION_PATH):
        os.makedirs(PREDICTION_PATH)

    if not os.path.exists(ERRORS_PATH):
        os.makedirs(ERRORS_PATH)

    if not os.path.exists(ERRORS_AUTOCOR_PATH):
        os.makedirs(ERRORS_AUTOCOR_PATH)

    if not os.path.exists(ERRORS_ERRORS_PATH):
        os.makedirs(ERRORS_ERRORS_PATH)

    if not os.path.exists(IMPROVEMENT_PATH):
        os.makedirs(IMPROVEMENT_PATH)

    if not os.path.exists(COMPARED_IMPROVEMENTS_PATH):
        os.makedirs(COMPARED_IMPROVEMENTS_PATH)


def add_features(data, features_conf):
    data = add_moments(data, features_conf.get('moments'))
    data = add_min_max(data, features_conf.get('min-max'))
    data = add_shmu_error(data, features_conf.get('shmu-error'))

    data = shmu_error_prediction_time_moment(
        data, features_conf.get('shmu-error-moment'))

    data = add_morning_and_afternoon_temp(
        data, features_conf.get('afternoon-morning'))

    conf = features_conf.get('shmu-error-p-time')
    if (conf):
        lags = conf.get('lags')
        lag_by = conf.get('lag_by')
        exp = conf.get('exp')
        data = shmu_prediction_time_error(data, lags, lag_by, exp)

    conf = features_conf.get('feature-lagged-p-time')
    if (conf):
        for c in conf:
            lags = c.get('lags')
            lag_by = c.get('lag_by')
            name = c.get('name')
            data = feature_lagged_by_hours_p_time(data, name, lags, lag_by)

    conf = features_conf.get('feature-lagged')
    if (conf):
        for c in conf:
            lags = c.get('lags')
            lag_by = c.get('lag_by')
            name = c.get('name')
            data = feature_lagged_by_hours(data, name, lags, lag_by)

    return data


if __name__ == '__main__':
    setup_env()
    data_path = config['data']
    data = pd.read_csv(data_path, delimiter=';')
    predictions_all = None

    for index, c in enumerate(config['models']):
        color_print('\nUsing {} model ...'.format(index + 1))

        transformed_data = data.copy(deep=True)
        transformed_data = add_features(transformed_data, c['features'])

        y = transformed_data.future_temp.values
        x = transformed_data.drop(fieldsToDrop, axis=1).values

        model, x = get_model(c['model'], c['model_params'], x)
        predicted_values = predict(
            data=transformed_data, x=x, y=y, model=model,
            window_length=c['window_length'],
            window_period=c['window_period'],
            weight=c.get('weight'),
            scale=c.get('scale'),
            autocorrect=c.get('autocorrect'),
            stable=config['stable']['active'],
            stable_func=config['stable']['func'],
            ignore_diff_errors=config['stable']['ide'],
            autocorrect_only_stable=config['stable']['aos'],
            diff=c.get('diff'),
            skip=c.get('skip'))

        predictions_all = merge_predictions(predictions_all, predicted_values)

    predictions_all_cleared = predictions_all.dropna()
    predicted_values = predictions_all_cleared.loc[
        :, predictions_all_cleared.columns != 'validity_date'].mean(axis=1)

    print('Merging predictions ...')
    final_values = join_date_and_values(predicted_values,
                                        predictions_all_cleared.validity_date)

    result = merge_with_measured_and_shmu_predictions(data, final_values)
    save_improvements(result)
    show_metrics(result)
    save_predictions(result)
    save_errors(result)

# TODO control calculations ... MAE, MSE
# TODO move parser documentation to conf.py
