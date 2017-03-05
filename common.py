import matplotlib.pyplot as plt
import numpy as np
import argparse
import math


def get_bias(real, predicted):
    return np.mean(real - predicted)


def show_predictions(real_values, predicted_values, shmu_predictions):
    plt.figure(1)
    plt.plot(real_values, 'ok', label='Real values')
    plt.plot(predicted_values, 'or', label='Predicted values (Our model)')
    plt.plot(shmu_predictions, 'og', label='Predicted values (SHMU)')
    plt.legend(loc=1)
    plt.title('Temperature predictions')
    plt.ylabel('Temperature')
    plt.xlabel('Samples')
    plt.show()


def show_errors(predicted_errors, shmu_errors):
    plt.figure(2)
    plt.plot(predicted_errors, 'k', label='predicted errors')
    plt.plot(shmu_errors, 'r', label='shmu errors')
    plt.legend(loc=1)
    plt.title('Temperature errors')
    plt.ylabel('Error')
    plt.xlabel('Samples')
    # plt.ylim((-30, 30))
    plt.show()


def get_parser():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--weight', action='store', dest='weight_coef',
                        help='''Weight coefficient. If none supplied, no
weights will be used.''', type=float)
    parser.add_argument('--file', action='store', dest='data_file',
                        required=True,
                        help='''Path to data file that will be loaded.''')
    parser.add_argument('--mode', action='store', dest='mode',
                        default='window',
                        choices=['window', 'extended-window', 'train-set'],
                        help="""Mode to use for predictions:\n
window = use sliding window
extended-window = use window that grows over time
train-set = learn from fixed length train set\n
Default length for window, extended-window and train-set is 60.
To override it set '--length' option.""")
    parser.add_argument('--length', action='store', dest='length',
                        default=60,
                        help='Length of window, extended-window or train-set.')
    parser.add_argument('--no-intercept', action='store_true', default=False,
                        dest='intercept',
                        help='If set will not use bias term.')
    return parser


def get_mode_action(mode):
    '''
    Return function to execure given mode.
    '''
    return modes_actions[mode]


def window(data, x, y, weight, lr, window_len, slide=False):
    data_len = x.shape[0]
    predictions_count = data_len - window_len
    mae_predict = 0
    mse_predict = 0
    mae_shmu = 0
    mse_shmu = 0
    start = 0
    predicted_all = np.array([])
    train_end = window_len

    # Check if all indexes are ok
    # check if data ok, previous work had some problem with wind or
    # other feature

    while (train_end < data_len):
        x_train = x.iloc[start:train_end, :]
        y_train = y.iloc[start:train_end]

        # Check how many prediction we can make within same ref_date
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

        lr.fit(x_train, y_train, sample_weight=weights)

        # predict values for y
        y_predicted = lr.predict(x_test)

        # add into predicted all
        predicted_all = np.hstack((predicted_all, y_predicted))

        # -1 index stands for current_temperature column in data
        mae_shmu += np.sum(abs(y_test - x_test.future_temp_shmu))
        mse_shmu += np.sum((y_test - x_test.future_temp_shmu) ** 2)

        mae_predict += np.sum(abs(y_test - y_predicted))
        mse_predict += np.sum((y_test - y_predicted) ** 2)

        # shift interval for learning
        train_end += pred_length
        if (slide):
            start += pred_length

    return {
        'mae_predict': mae_predict / predictions_count,
        'mae_shmu': mae_shmu / predictions_count,
        'mse_predict': mse_predict / predictions_count,
        'mse_shmu': mse_shmu / predictions_count,
        'predicted_all': np.array(predicted_all),
        'predictions_count': predictions_count,
    }


def _sliding_window(data, x, y, weight, lr, window_len):
    return window(data, x, y, weight, lr, window_len, slide=True)


def _extended_window(data, x, y, weight, lr, window_len):
    return window(data, x, y, weight, lr, window_len, slide=False)


def _train_set_approach():
    pass

modes_actions = {
    'window': _sliding_window,
    'extended-window': _extended_window,
    'train-set': _train_set_approach
}
