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

from utils import save_predictions, save_bias
from utils import save_errors, predict_new
from parsers import get_predict_parser

from conf import config

# The pandas warning is statsmodel issue


class Colors:
    BLUE = '\033[94m'
    ENDC = '\033[0m'


def color_print(text, color=Colors.BLUE):
    print(color + text + Colors.ENDC)

'''
Basic model
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


def get_model(name, params):
    model = None
    if (name == 'ols'):
        model = lm.LinearRegression(fit_intercept=params['fit_intercept'])
    return model


def merge_predictions(predictions_all, predicted_values):
    if (predictions_all is not None):
        return pd.merge(predictions_all, predicted_values,
                        on='validity_date', how='outer')
    return predicted_values


def get_final_values(predicted_values):
    avg = []
    for i in predicted_values.index:
        avg.append(np.mean(predicted_values.loc[i, :].values))

    fin = pd.DataFrame()
    fin['validity_date'] = predictions_all_cleared.validity_date
    fin['predicted'] = pd.Series(avg)
    return fin


def show_metrics(data, final_predictions):
    cols_to_pick = ['validity_date', 'future_temp', 'future_temp_shmu']
    merged = pd.merge(final_predictions, data.loc[:, cols_to_pick],
                      on='validity_date', how='inner').dropna()

    mae_model = mean_absolute_error(merged.future_temp, merged.predicted)
    mse_model = mean_squared_error(merged.future_temp, merged.predicted)

    mae_shmu = mean_absolute_error(
        merged.future_temp, merged.future_temp_shmu)
    mse_shmu = mean_squared_error(
        merged.future_temp, merged.future_temp_shmu)

    color_print('Model')
    print('MAE: {0:.4f}'.format(mae_model))
    print('MSE: {0:.4f}'.format(mse_model))

    color_print('SHMU')
    print('MAE: {0:.4f}'.format(mae_shmu))
    print('MSE: {0:.4f}'.format(mse_shmu))

    print('\nNumber of predictions', merged.shape[0])


def setup_env():
    pd.set_option('display.max_columns', None)

    if not os.path.exists('./other'):
        os.makedirs('./other')


if __name__ == '__main__':
    setup_env()
    data_path = config['data']
    data = pd.read_csv(data_path, delimiter=';')
    predictions_all = None

    for index, c in enumerate(config['models']):
        print('Using {} model ...'.format(index + 1))

        y = data.future_temp.values
        x = data.drop(fieldsToDrop, axis=1).values

        model = get_model(c['model'], c['model_params'])
        predicted_values = predict_new(data=data, x=x, y=y, model=model,
                                       window_length=c['window_length'],
                                       window_period=c['window_period'],
                                       weight=c.get('weight'),
                                       autocorrect=c.get('autocorrect'))
        predictions_all = merge_predictions(predictions_all, predicted_values)

    predictions_all_cleared = predictions_all.dropna()
    predicted_values = predictions_all_cleared.loc[
        :, predictions_all_cleared.columns != 'validity_date']

    print('Merging predictions ...')
    final_values = get_final_values(predicted_values)

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

    show_metrics(data, final_values)
