import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse
from os import sys


def get_bias(real, predicted):
    r = real.astype(float)
    p = predicted.astype(float)
    return np.mean(r - p)


def show_figure(real_values, predicted_values, shmu_predicted):
    plt.figure(1)
    plt.plot(real_values, 'ok', label='Real values')
    plt.plot(predicted_values, 'or', label='Predicted values (Our model)')
    plt.plot(shmu_predicted, 'og', label='Predicted values (SHMU)')
    plt.legend(loc=1)
    plt.title('Temperature predictions')
    plt.ylabel('Temperature')
    plt.xlabel('Samples')
    plt.show()


def load_data(path):
    data = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        i = 0
        for row in reader:
            if (row):
                if (i != 0):
                    data.append(row)
                i += 1
    return data


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', action='store', dest='weight_coef',
                        help='''Weight coefficient. If none suplied, no
                                weights will be used.''', type=float)
    parser.add_argument('-p', action='store', dest='data_path',
                        help='''Path to data file that will be loaded.''')
    parser.add_argument('--no-intercept', action='store_true', default=False,
                        dest='intercept',
                        help='If set want use bias term.')
    return parser


def check_args(args):
    if (not args.data_path):
        print('Path to data file has to be provided!')
        sys.exit(1)
    return args
