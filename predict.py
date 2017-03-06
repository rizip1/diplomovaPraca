import sklearn.linear_model as lm
import pandas as pd
import os

from common import get_bias, save_predictions, get_parser, get_mode_action
from common import save_errors


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

    data = pd.read_csv(args.data_file, delimiter=';')
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

    lr = lm.LinearRegression(fit_intercept=fit_intercept)

    mode_action = get_mode_action(mode)
    stats = mode_action(data, x, y, weight, lr, length)

    print('BIAS in data {0:.2f}'.format(get_bias(
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
