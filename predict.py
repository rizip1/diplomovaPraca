import sklearn.linear_model as lm
import pandas as pd

from common import get_bias, show_predictions, get_parser, get_mode_action
from common import show_errors


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    parser = get_parser()
    args = parser.parse_args()

    # load command line arguments
    weight = (args.weight_coef, None)[args.weight_coef is None]
    fit_intercept = (True, False)[args.intercept is None]
    mode = args.mode
    length = int(args.length)

    data = pd.read_csv(args.data_file, delimiter=';')

    # Shape info
    print('Shape', data.shape)

    # Columns info
    # print('columns', data.columns)

    y = data.future_temp

    fieldsToDrop = ['future_temp', 'validity_date', 'reference_date']
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

    show_predictions(real_values=data.future_temp,
                     predicted_values=stats['predicted_all'],
                     shmu_predictions=data.future_temp_shmu)

    show_errors(predicted_errors, shmu_errors)
'''
Linear regression coefficients will be identical if you do, or don't,
scale your data,
because it's looking at proportional relationships between them

but can run faster when using gradient descent
'''
