import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from constants import PREDICTION_PATH
from utils import color_print

'''
Except data containing 'validity_date', 'predicted' and 'future_temp' columns
'''


def get_parser():
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


def get_data():
    parser = get_parser()
    args = parser.parse_args()

    path1 = args.file1
    path2 = args.file2
    d1 = pd.read_csv('{}/{}'.format(PREDICTION_PATH, path1),
                     delimiter=';', header=0)
    d2 = pd.read_csv('{}/{}'.format(PREDICTION_PATH, path2),
                     delimiter=';', header=0)
    return (d1, d2)


def show_results(d1, d2, merged):
    mae_d1 = mae(d1.loc[index].predicted, d1.loc[index].future_temp)
    mse_d1 = mse(d1.loc[index].predicted, d1.loc[index].future_temp)

    mae_d2 = mae(d2.loc[index].predicted, d2.loc[index].future_temp)
    mse_d2 = mse(d2.loc[index].predicted, d2.loc[index].future_temp)

    color_print('Data1 stats')
    print('MAE {0:.4f}'.format(mae_d1))
    print('MSE {0:.4f}'.format(mse_d1))

    color_print('\nData2 stats')
    print('MAE {0:.4f}'.format(mae_d2))
    print('MSE {0:.4f}'.format(mse_d2))

    print('\nPredictions count {}'.format(merged.shape[0]))


if __name__ == '__main__':
    d1, d2 = get_data()

    merged = pd.merge(d1, d2, on='validity_date', how='inner')
    merged.set_index('validity_date', inplace=True)
    index = merged.index  # can query common rows using common index

    d1.set_index('validity_date', inplace=True)
    d2.set_index('validity_date', inplace=True)

    show_results(d1, d2, merged)
