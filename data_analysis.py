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
    parser.add_argument('--skip-data', action='store_true', default=False,
                        dest='skip_data',
                        help='Will not recreate data files')
    parser.add_argument('--skip-invalid', action='store_true', default=False,
                        dest='skip_invalid',
                        help='Will not recreate invalid data')
    parser.add_argument('--skip-important', action='store_true', default=False,
                        dest='skip_important',
                        help='Will not recreate most important features')
    parser.add_argument('--skip-hists', action='store_true', default=False,
                        dest='skip_hists',
                        help='Will not recreate hists files')
    return parser


def shift_invalid_values(x, mean, column):
    if (x == -999):
        if (column == 'pressure'):
            return 200000
        if (column == 'current_temp' or column == 'future_temp' or
                column == 'future_temp_shmu'):
            return 60
        if (column == 'humidity'):
            return 200
        if (column == 'wind_direction'):
            return 600
        if (column == 'wind_speed'):
            return 100
        if (column == 'rainfall_last_hour'):
            return 50
        return mean * 2
    return x


def save_invalid_data_to_csv(filename, invalid_rows_counts_all):
    with open('{}.csv'.format(filename), 'w') as csvfile:
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


def save_invalid_data_to_plots(folder, colors, invalid_rows_all):
    shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)
    for station, i_data in invalid_rows_all.items():
        print('Saving plot with missing data for station {}...'
              .format(station))
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        i = 0
        for label, value in i_data.items():
            plt.plot(value, 'o', color=colors[i], label=label)
            i += 1

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
        plt.title('Missing data for station {}'.format(station))
        plt.ylabel('Features')
        plt.xlabel('Samples')
        plt.savefig('missing_data/{}.png'.format(station))
        plt.close(fig)


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


def get_most_important_features(x, y, out_file):
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
    plt.savefig(out_file)
    plt.close()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    skip_data = args.skip_data
    skip_invalid = args.skip_invalid
    skip_important = args.skip_important
    skip_hists = args.skip_hists

    stations = get_stations()

    # query and save data
    if (not skip_data):
        shutil.rmtree('./data/', ignore_errors=True)
        os.mkdir('./data')
        for s in stations:
            print('Processing station {}...'.format(s))
            save_data_for_station(
                station_id=s, out_file='data/data_{}.csv'.format(s))

    # Get invalid data
    if (not skip_invalid):
        invalid_rows_all = {}
        invalid_rows_counts_all = {}
        for s in stations:
            print('Processing invalid data for station {}'.format(s))
            data = pd.read_csv('data/data_{}.csv'.format(s), delimiter=';')
            invalid_rows, invalid_rows_counts = get_invalid_rows(data)
            invalid_rows_all[s] = invalid_rows
            invalid_rows_counts_all[s] = invalid_rows_counts

        save_invalid_data_to_csv(
            filename='invalid_rows',
            invalid_rows_counts_all=invalid_rows_counts_all)

        colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c', '#8080A0']
        save_invalid_data_to_plots(folder='./missing_data/', colors=colors,
                                   invalid_rows_all=invalid_rows_all)

    # Get most important features based on RandomForest training
    if (not skip_important):
        shutil.rmtree('./other/', ignore_errors=True)
        os.mkdir('./other/')
        print('Getting feature importance ...')
        complete_station = 11894
        complete_station_data = pd.read_csv(
            'data/data_{}.csv'.format(complete_station), delimiter=';')
        y = data.future_temp
        fieldsToDrop = ['future_temp', 'validity_date', 'reference_date']
        x = data.drop(fieldsToDrop, axis=1)
        get_most_important_features(
            x, y, out_file='./other/feature_importance.png')
        print('Finished getting feature importance')

    if (not skip_hists):
        shutil.rmtree('./hists/', ignore_errors=True)
        os.mkdir('./hists/')
        os.mkdir('./hists/complete_hists/')
        for s in stations:
            print('Creating hists for station {}'.format(s))
            data = pd.read_csv('data/data_{}.csv'.format(s), delimiter=';')

            ignore = ['validity_date', 'reference_date']
            for c in data.columns:
                if (c not in ignore):
                    data[c] = data[c].apply(
                        shift_invalid_values, args=(abs(data[c].mean()), c))

            # histogram for all features in one figure
            data.hist(figsize=(35, 15), bins=100)
            plt.savefig('hists/complete_hists/{}.png'.format(s))
            plt.close()

            os.mkdir('hists/{}/'.format(s))

            d = data.drop(ignore, axis=1)
            # histogram for separate features
            for c in d.columns:
                ax = d.hist(c, figsize=(20, 15), bins=100)
                plt.savefig('hists/{}/{}.png'.format(s, c))
                plt.close()
