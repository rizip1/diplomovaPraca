import matplotlib.pyplot as plt
import shutil
import csv
import os
import numpy as np
import pandas as pd
import argparse

from db import save_data_for_station, get_stations
from sklearn.ensemble import ExtraTreesRegressor


def get_parser():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--cache', action='store_true', default=False,
                        dest='cache',
                        help='Will not recreate data files')
    return parser


def wind_speed_scale(x):
    if (x < 0):
        return 2000
    else:
        return x * 100


def remove_undef(x):
    if (x >= 0):
        return x


def get_invalid_rows(data):
    '''
    Return dictionary containing records with invalid
    values where keys are column names and values
    are lists containing zeros and positive integers.
    Zeros flags indicate that value is OK and positive
    integers indicate invalid value. To each column
    is assigned different error integer. Also return
    dictionary containing counts of invalid values for
    each column separately.
    '''

    invalid_rows = {}
    invalid_rows_counts = {}
    ignore_columns = ['reference_date', 'validity_date']

    for index, column in enumerate(data.columns):
        if (column in ignore_columns):
            continue
        invalid_rows[column] = []
        values = data[column].values
        invalid_rows_counts[column] = 0
        for value in values:
            if (value == -999):
                invalid_rows[column].append(index + 1)
                invalid_rows_counts[column] += 1
            else:
                invalid_rows[column].append(0)

    return (invalid_rows, invalid_rows_counts)


def get_most_important_features(x, y):
    forest = ExtraTreesRegressor(n_estimators=250,
                                 random_state=0)
    _x = x.values
    _y = y.values
    forest.fit(_x, _y)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(25, 15))
    plt.title('Feature importances')
    plt.bar(range(x.shape[1]), importances[indices],
            color='r', yerr=std[indices], align='center')
    plt.xticks(range(x.shape[1]), np.array(x.columns)[indices],
               rotation='vertical')
    plt.xlim([-1, x.shape[1]])
    plt.savefig('{}.png'.format('feature_importances'))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    cache = args.cache
    '''
    data['wind_speed_scaled'] = data['wind_speed'].apply(wind_speed_scale)

    # for Random Forest dimensions must match
    data['wind_speed'] = data['wind_speed'].apply(remove_undef)
    data['wind_direction'] = data['wind_direction'].apply(remove_undef)

    print('var', data['wind_speed'].var())
    print('mean', data['wind_speed'].mean())

    data.hist(figsize=(35, 15), bins=100)
    plt.savefig('{}.png'.format('sss'))

    y = data.future_temp
    fieldsToDrop = ['future_temp', 'validity_date', 'wind_speed',
                    'wind_direction', 'reference_date',
                    'wind_speed_scaled']
    x = data.drop(fieldsToDrop, axis=1)
    # get_most_important_features(x, y)

    ignore = ['validity_date', 'reference_date']

    d = data.drop(ignore, axis=1)
    for c in d.columns:
        print('c', type(d[c]), c)
        ax = d.hist(c, figsize=(20, 15), bins=100)
        # fig = ax.get_figure()
        plt.savefig('hists/{}.png'.format(c))
    '''
    stations = get_stations()

    if (not cache):
        shutil.rmtree('./data/')
        os.mkdir('./data')
        for s in stations:
            print('Processing station {}...'.format(s))
            save_data_for_station(
                station_id=s, out_file='data/data_{}.csv'.format(s))

    invalid_rows_all = {}
    invalid_rows_counts_all = {}
    for s in stations:
        data = pd.read_csv('data/data_{}.csv'.format(s), delimiter=';')
        invalid_rows, invalid_rows_counts = get_invalid_rows(data)
        invalid_rows_all[s] = invalid_rows
        invalid_rows_counts_all[s] = invalid_rows_counts

    with open('invalid_rows.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = False
        invalid_rows_sorted = sorted(list(invalid_rows_counts_all.items()))
        for key, data_record in invalid_rows_sorted:
            if (not header):
                column_names = sorted(list(data_record.keys()))
                writer.writerow(['station'] + column_names)
                header = True

            row = []
            row.append(key)
            for key in sorted(list(data_record.keys())):
                row.append(data_record[key])
            writer.writerow(row)

    # TODO histogram for each feature separately

    complete_station = 11894
    complete_station_data = pd.read_csv(
        'data/data_{}.csv'.format(complete_station), delimiter=';')
    y = data.future_temp
    fieldsToDrop = ['future_temp', 'validity_date', 'reference_date']
    x = data.drop(fieldsToDrop, axis=1)
    get_most_important_features(x, y)

    # random forest feature importances for 'wind_speed' and
    # 'wind_direction' for continuous period (find that period)

    # Also check other stations missing data
    # Check data as whole
