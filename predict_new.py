import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.ensemble as dt
import sklearn.neural_network as nn
import sklearn.neighbors as ng
import pandas as pd
import os

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures

from feature_utils import add_moments
from feature_utils import shmu_error_prediction_time_moment
from feature_utils import feature_lagged_by_hours
from feature_utils import feature_lagged_by_hours_p_time
from feature_utils import shmu_prediction_time_error
from feature_utils import add_shmu_error
from feature_utils import add_min_max
from feature_utils import add_morning_and_afternoon_temp

# from utils import save_predictions, save_bias, save_errors
from utils import color_print, predict_new
from parsers import get_predict_parser

from conf import config

# The pandas warning is statsmodel issue

from constants import PREDICTION_PATH, OTHER_PATH


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
        model = lm.LassoCV(fit_intercept=True, normalize=True, cv=cv)
    elif (name == 'poly-lasso'):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        x = poly.fit_transform(x)
        model = lm.Lasso(alpha=0.3, copy_X=True, fit_intercept=True,
                         normalize=False)
    elif (name == 'ridge'):
        # higher alpha = more regularization
        model = lm.Ridge(alpha=0.5, copy_X=True, fit_intercept=True,
                         normalize=False)
    elif (name == 'ridge-cv'):
        cv = TimeSeriesSplit(n_splits=5)
        model = lm.RidgeCV(fit_intercept=True, normalize=False, cv=cv)
    elif (name == 'elastic-cv'):
        cv = TimeSeriesSplit(n_splits=5)
        model = lm.ElasticNetCV(cv=cv)
    elif (name == 'bayes-ridge'):
        model = lm.BayesianRidge()
    elif (name == 'svr'):
        model = svm.SVR(kernel='linear', C=1000)
        # model = svm.SVR(kernel='rbf', C=1000, gamma=0.05)
    elif (name == 'knn'):
        model = ng.KNeighborsRegressor(n_neighbors=1)
    return model


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
    result[['validity_date', 'predicted', 'future_temp']].to_csv(
        '{}/predictions.csv'.format(PREDICTION_PATH),
        index=False, sep=';')


def merge_with_measured(data, final_predictions):
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


if __name__ == '__main__':
    setup_env()
    data_path = config['data']
    data = pd.read_csv(data_path, delimiter=';')
    predictions_all = None

    for index, c in enumerate(config['models']):
        print('Using {} model ...'.format(index + 1))

        y = data.future_temp.values
        x = data.drop(fieldsToDrop, axis=1).values

        model = get_model(c['model'], c['model_params'], x)
        predicted_values = predict_new(
            data=data, x=x, y=y, model=model,
            window_length=c['window_length'],
            window_period=c['window_period'],
            weight=c.get('weight'),
            autocorrect=c.get('autocorrect'),
            stable=config['stable']['active'],
            stable_func=config['stable']['func'],
            ignore_diff_errors=config['stable']['ide'],
            autocorrect_only_stable=config['stable']['aos'])

        predictions_all = merge_predictions(predictions_all, predicted_values)

    predictions_all_cleared = predictions_all.dropna()
    predicted_values = predictions_all_cleared.loc[
        :, predictions_all_cleared.columns != 'validity_date'].mean(axis=1)

    print('Merging predictions ...')
    final_values = join_date_and_values(predicted_values,
                                        predictions_all_cleared.validity_date)

    # TODO normalization (at least for SVR)
    # TODO stable weather detection
    # TODO adding new features
    # TODO skip predictions
    # TODO improvements
    # TODO hour results
    # TODO normality
    # TODO bias
    # TODO norm
    # TODO diff
    # TODO caching
    # TODO directories checking
    # TODO refactor

    result = merge_with_measured(data, final_values)
    show_metrics(result)
    save_predictions(result)
