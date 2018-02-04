import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from constants import PREDICTION_PATH
from utils import color_print
from improvements import compare_2_models_improvements
from scipy.stats import wilcoxon, kstest, norm
from constants import OTHER_PATH

'''
Except data containing 'validity_date', 'predicted' and 'future_temp' columns
'''


def _get_parser():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--file1', action='store', dest='file1',
                        required=True,
                        help='Path to first file')
    parser.add_argument('--file2', action='store', dest='file2',
                        required=True,
                        help='Path to second file')
    return parser


def _get_data():
    parser = _get_parser()
    args = parser.parse_args()

    path1 = args.file1
    path2 = args.file2
    d1 = pd.read_csv('{}/{}'.format(PREDICTION_PATH, path1),
                     delimiter=';', header=0)
    d2 = pd.read_csv('{}/{}'.format(PREDICTION_PATH, path2),
                     delimiter=';', header=0)
    return (d1, d2)


def _show_results(d1, d2):
    mae_d1 = mae(d1.predicted, d1.future_temp)
    mse_d1 = mse(d1.predicted, d1.future_temp)

    mae_d2 = mae(d2.predicted, d1.future_temp)
    mse_d2 = mse(d2.predicted, d2.future_temp)

    color_print('Data1 stats')
    print('MAE {0:.4f}'.format(mae_d1))
    print('MSE {0:.4f}'.format(mse_d1))

    color_print('\nData2 stats')
    print('MAE {0:.4f}'.format(mae_d2))
    print('MSE {0:.4f}'.format(mse_d2))

    print('\nPredictions count {}'.format(d1.shape[0]))

    err1 = np.absolute(d1.predicted - d1.future_temp)
    err2 = np.absolute(d2.predicted - d2.future_temp)

    dist = err1 - err2
    mu, std = norm.fit(dist)
    # tests the null hypothesis that values (absolute errors)
    # commes from normal distribution
    print('Kol-Schirnov', kstest((dist - mu) / std, 'norm').pvalue)

    # plot absolute errors dependence
    plt.figure(figsize=(12, 6))
    plt.plot(err1, err2, 'o')
    plt.title('Absolute errors dependence')
    plt.ylabel('Errors1')
    plt.xlabel('Errors2')
    plt.savefig('{}/errors_dependance.png'.format(OTHER_PATH))

    # tests the null hypothesis that two related paired
    # samples come from the same distribution
    print('Paired Wilcoxon', wilcoxon(err1, err2).pvalue)


def _save_improvements(d1, d2):
    compare_2_models_improvements(d1, d2)


if __name__ == '__main__':
    d1, d2 = _get_data()

    merged = pd.merge(d1, d2, on='validity_date', how='inner')
    merged.set_index('validity_date', inplace=True)
    index = merged.index  # can query common rows using common index

    # validity_date will be index and will not exist as column
    d1.set_index('validity_date', inplace=True)
    d2.set_index('validity_date', inplace=True)

    d1_common = d1.loc[index]
    d2_common = d2.loc[index]
    _show_results(d1_common, d2_common)

    # get validity_column back
    d1_common.reset_index(level=0, inplace=True)
    d2_common.reset_index(level=0, inplace=True)
    _save_improvements(d1_common, d2_common)
