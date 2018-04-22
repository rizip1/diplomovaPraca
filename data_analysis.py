import matplotlib.pyplot as plt
import shutil
import csv
import os
import numpy as np
import pandas as pd
import argparse
import re
from datetime import datetime, timedelta

from db import save_data_for_station, get_stations
from sklearn.ensemble import ExtraTreesRegressor

pd.options.mode.chained_assignment = None

label_mapping = {
    'current_pressure': 'Pressure',
    'current_wind_speed': 'Wind speed',
    'current_wind_direction': 'Wind direction',
    'current_rainfall_last_hour': 'Rainfall',
    'current_temp': 'Temperature',
    'current_humidity': 'Humidity',
    'future_temp_shmu': 'Aladin temperature',
    'p_time_pressure': 'Pressure',
    'p_time_wind_speed': 'Wind speed',
    'p_time_wind_direction': 'Wind direction',
    'p_time_rainfall_last_hour': 'Rainfall',
    'p_time_temp': 'Temperature',
    'p_time_humidity': 'Humidity',
}
label_order = ['current_pressure', 'current_wind_speed',
               'current_wind_direction', 'current_rainfall_last_hour',
               'current_temp', 'current_humidity', 'future_temp_shmu']

'''
TODO refactor this file
'''


def get_parser():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', action='store_true', default=False,
                        dest='data',
                        help='Will recreate data files')
    parser.add_argument('--data-missing', action='store', default=False,
                        dest='data_missing_path',
                        help='''Will check missing records. Argument is name
of the data folder where to find data files.''')
    parser.add_argument('--invalid', action='store', default=False,
                        dest='invalid',
                        help='''Will recreate invalid data. Argument
is input_folder-output_folder''')
    parser.add_argument('--important', action='store_true', default=False,
                        dest='important',
                        help='Will recreate most important features')
    parser.add_argument('--hists', action='store_true', default=False,
                        dest='hists',
                        help='Will recreate hists files')
    parser.add_argument('--corr', action='store_true', default=False,
                        dest='corr',
                        help='Will recreate correlation matrix')
    parser.add_argument('--features', action='store_true', default=False,
                        dest='features',
                        help='Will plot features in time')
    parser.add_argument('--shmu-errors', action='store_true', default=False,
                        dest='shmu_errors',
                        help='Will plot shmu errors in time')
    parser.add_argument('--stable-weather', action='store_true', default=False,
                        dest='stable_weather',
                        help='Will plot table weather')
    parser.add_argument('--replace-missing', action='store_true',
                        default=False, dest='replace_missing',
                        help='Will replace missing data')
    return parser


def shift_invalid_values(x, mean, column):
    if (x == -999):
        if (column == 'current_pressure' or column == 'p_time_pressure'):
            return 200000
        if (column == 'current_temp' or column == 'future_temp' or
                column == 'future_temp_shmu' or column == 'p_time_temp'):
            return 60
        if (column == 'current_humidity' or column == 'p_time_humidity'):
            return 200
        if (column == 'current_wind_direction' or
                column == 'p_time_wind_direction'):
            return 600
        if (column == 'current_wind_speed' or column == 'p_time_wind_speed'):
            return 100
        if (column == 'current_rainfall_last_hour' or
                column == 'p_time_rainfall_last_hour'):
            return 50
        return mean * 2
    return x


def save_invalid_data_to_csv(filename, invalid_rows_counts_all, observations):
    with open('{}.csv'.format(filename), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = False
        invalid_rows_sorted = sorted(list(invalid_rows_counts_all.items()))
        for key, data_record in invalid_rows_sorted:
            if (not header):
                column_names = sorted(list(data_record.keys()))
                writer.writerow(['station', 'n_observations'] + column_names)
                header = True

            row = []
            row.append(key)
            row.append(observations[key])
            for key in sorted(list(data_record.keys())):
                row.append(data_record[key])
            writer.writerow(row)


def plot_features(data):
    folder = 'features'
    shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)
    for c in data.columns:
        os.mkdir('{}/{}'.format(folder, c))
        # ignore errors for strings features like date
        fig = plt.figure(figsize=(10, 6))
        try:
            plt.plot(data[c])
            plt.savefig('{}/{}/orig.png'.format(folder, c))
            plt.close(fig)
        except:
            plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        try:
            diff = data[c].diff()
            plt.plot(diff)
            plt.savefig('{}/{}/diff_1.png'.format(folder, c))
            plt.close(fig)
        except:
            plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        try:
            diff = data[c].diff(periods=12)
            plt.plot(diff)
            plt.savefig('{}/{}/diff_12.png'.format(folder, c))
            plt.close(fig)
        except:
            plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        try:
            diff = data[c].diff(periods=24)
            plt.plot(diff)
            plt.savefig('{}/{}/diff_24.png'.format(folder, c))
            plt.close(fig)
        except:
            plt.close(fig)


def create_correlation_matrix(data, out_file):
    corr = data.corr()
    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, interpolation='nearest')
    fig.colorbar(cax)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.06)

    columns = []
    for c in corr.columns:
        columns.append(label_mapping[c])

    plt.xticks(range(len(corr.columns)), columns,
               rotation='vertical', fontsize=14)
    plt.yticks(range(len(corr.columns)), columns, fontsize=14)
    plt.savefig('{}/{}.png'.format('other', out_file))


def save_invalid_data_to_plots(folder, colors, invalid_rows_all, output):
    shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)
    for station, i_data in invalid_rows_all.items():
        print('Saving plot with missing data for station {}...'
              .format(station))
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        i = 0
        for label in label_order:
            x = []
            y = []
            inv_data = i_data[label]
            for pos, value in enumerate(inv_data):
                if (value != 0):
                    x.append(pos)
                    y.append(value)

            plt.plot(x, y, 'o', color=colors[i], label=label_mapping[label])
            i += 1

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
        plt.title('Invalid data for station {}'.format(station))
        plt.ylabel('Features')
        plt.xlabel('Samples')
        plt.yticks([f for f in range(1, 8)])
        plt.savefig('{}/{}.png'.format(output, station))
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
    forest = ExtraTreesRegressor(n_estimators=500,
                                 random_state=0)
    _x = x.values
    _y = y.values
    forest.fit(_x, _y)

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)

    plt.figure(figsize=(20, 10))
    plt.title('Features importances', fontsize=20)
    plt.bar(range(x.shape[1]), importances[indices],
            color='r', align='center', yerr=std[indices])

    columns = []
    for c in x.columns:
        columns.append(label_mapping[c])

    plt.xticks(range(x.shape[1]), np.array(columns)[indices],
               rotation='horizontal', fontsize=14)
    plt.xlim([-1, x.shape[1]])
    plt.savefig(out_file)
    plt.close()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    create_data = args.data
    create_invalid = args.invalid
    invalid_in, invalid_out = [None, None]
    if (create_invalid):
        invalid_in, invalid_out = create_invalid.split('-')
    create_important = args.important
    create_hists = args.hists
    create_corr = args.corr
    create_features = args.features
    data_missing_path = args.data_missing_path
    create_shmu_errors = args.shmu_errors
    stable_weather = args.stable_weather
    replace_missing = args.replace_missing

    stations = get_stations()

    # query and save data
    if (create_data):
        shutil.rmtree('./data/', ignore_errors=True)
        os.mkdir('./data')
        for s in stations:
            prev_hour = None
            print('Processing station {}...'.format(s))
            save_data_for_station(
                station_id=s, out_file='data/data_{}.csv'.format(s))

    if (replace_missing):
        hour_regex = r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$'
        day_regex = r'^([0-9]{4}-[0-9]{2}-[0-9]{2}) [0-9]{2}:[0-9]{2}:[0-9]{2}$'
        for s in stations:
            print('Processing station {} ...'.format(s))
            s_data = pd.read_csv(
                'data/data_{}.csv'.format(s), delimiter=';')

            missing_v = -999
            sec_hour = 3600
            columns = s_data.columns
            start = s_data.loc[0, 'validity_date']
            end = s_data.loc[s_data.shape[0] - 1, 'validity_date']
            date_format = '%Y-%m-%d %H:%M:%S'

            date_1 = datetime.strptime(start, date_format)
            date_2 = datetime.strptime(end, date_format)
            delta = date_2 - date_1
            all_data_len = (delta.days * 24) + (delta.seconds // sec_hour)

            rows = []
            for j in range(s_data.shape[0] - 1):
                current_row = s_data.loc[j].values
                rows.append(current_row)
                ref_date1 = s_data.loc[j, 'reference_date']
                val_date1 = s_data.loc[j, 'validity_date']
                val_date2 = s_data.loc[j + 1, 'validity_date']
                d1 = datetime.strptime(val_date1, date_format)
                d2 = datetime.strptime(val_date2, date_format)
                d = d2 - d1

                if (d.seconds != sec_hour or d.days > 0):
                    miss_c = (d.days * 24) + (d.seconds // sec_hour) - 1

                    for m in range(1, miss_c + 1):
                        new_record = pd.Series()

                        new_val_date = str(datetime.strptime(
                            val_date1, date_format) + timedelta(hours=m))

                        match = re.search(hour_regex, new_val_date)
                        hour = int(match.group(1))
                        new_ref_date = None
                        match = re.search(day_regex, new_val_date)
                        ref_date_base = match.group(1)
                        if (hour >= 13 or hour == 0):
                            new_ref_date = '{} 12:00:00'.format(ref_date_base)
                        else:
                            new_ref_date = '{} 00:00:00'.format(ref_date_base)

                        # order must be preserved
                        new_record['reference_date'] = new_ref_date
                        new_record['validity_date'] = new_val_date

                        prev_index = j + m - 24
                        next_index = j + m + 24 - miss_c

                        # if this is true we can average records
                        if (prev_index >= 0 and next_index < s_data.shape[0] and miss_c < 24):
                            prev_r = s_data.loc[prev_index]
                            next_r = s_data.loc[next_index]

                            for c in columns:
                                # if one value is -999 we have to set
                                # other to -999 also so mean is again -999
                                if (prev_r[c] == missing_v or
                                        next_r[c] == missing_v):
                                    prev_r[c] = missing_v
                                    next_r[c] = missing_v
                                # take average of t-24 and t+24
                                if (c != 'validity_date' and
                                        c != 'reference_date'):
                                    new_record[c] = (prev_r[c] + next_r[c]) / 2
                        else:
                            # just add row of all values set to -999
                            for c in columns:
                                if (c != 'validity_date' and
                                        c != 'reference_date'):
                                    new_record[c] = -999
                        rows.append(new_record.values)

            # last record that we have
            rows.append(s_data.loc[s_data.shape[0] - 1].values)

            # when all missing data are replace, can replace invalid values
            for i in range(len(rows)):
                # skip reference_date and validity date
                for j in range(2, len(rows[i])):
                    if (rows[i][j] == missing_v and i >= 24 and
                            i + 24 < len(rows)):
                        if (not (rows[i - 24][j] == missing_v or
                                 rows[i + 24][j] == missing_v)):
                            rows[i][j] = (rows[i - 24][j] +
                                          rows[i + 24][j]) / 2

            new_data = pd.DataFrame(np.array(rows), columns=columns)
            new_data.to_csv('new_data/data_{}.csv'.format(s),
                            index=False, sep=';')

    if (data_missing_path):
        results = {}
        for s in stations:
            print('Processing station {}...'.format(s))
            total_missing_data = 0
            date_format = '%Y-%m-%d %H:%M:%S'
            s_data = pd.read_csv(
                '{}/data_{}.csv'.format(data_missing_path, s), delimiter=';')
            for j in range(0, s_data.shape[0] - 1):
                val_date1 = s_data.loc[j, 'validity_date']
                val_date2 = s_data.loc[j + 1, 'validity_date']
                d1 = datetime.strptime(val_date1, date_format)
                d2 = datetime.strptime(val_date2, date_format)
                delta = d2 - d1
                if (delta.seconds != 3600 or delta.days > 0):
                    miss_c = 0
                    if (delta.days > 0):
                        miss_c += delta.days * 24
                    if (delta.seconds > 3600):
                        miss_c += (delta.seconds // 3600) - 1
                    total_missing_data += miss_c
                    print('Missing data at pos {} for {} observations'.format(
                        j, miss_c))
                results[s] = total_missing_data

        for s in stations:
            print('{}: Total missing data count = {}'.format(s, results[s]))

    complete_station = 11816  # however no records about rainfall
    complete_station_data = pd.read_csv(
        'data/data_{}.csv'.format(complete_station), delimiter=';')
    complete_station_data = complete_station_data.iloc[0:13000]
    fieldsToDrop = ['future_temp', 'validity_date', 'reference_date',
                    'p_time_temp', 'p_time_wind_speed',
                    'p_time_wind_direction', 'p_time_humidity',
                    'p_time_rainfall_last_hour', 'p_time_pressure']

    # Get invalid data
    if (create_invalid):
        invalid_rows_all = {}
        invalid_rows_counts_all = {}
        observations = {}
        for s in stations:
            print('Processing invalid data for station {}'.format(s))
            data = pd.read_csv(
                '{}/data_{}.csv'.format(invalid_in, s), delimiter=';')
            data = data.drop(fieldsToDrop, axis=1)
            observations[s] = data.shape[0]
            invalid_rows, invalid_rows_counts = get_invalid_rows(data)
            invalid_rows_all[s] = invalid_rows
            invalid_rows_counts_all[s] = invalid_rows_counts

        save_invalid_data_to_csv(
            filename='invalid_rows',
            invalid_rows_counts_all=invalid_rows_counts_all,
            observations=observations)

        colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c', '#8080A0']
        save_invalid_data_to_plots(folder='./{}/'.format(invalid_out),
                                   colors=colors,
                                   invalid_rows_all=invalid_rows_all,
                                   output=invalid_out)

    # Get most important features based on RandomForest training
    if (create_important):
        print('Getting feature importance ...')

        fieldsToDrop = ['future_temp', 'validity_date', 'reference_date',
                        'current_temp', 'current_wind_speed',
                        'current_wind_direction', 'current_humidity',
                        'current_rainfall_last_hour', 'current_pressure']

        y = complete_station_data.future_temp
        x = complete_station_data.drop(fieldsToDrop, axis=1)
        get_most_important_features(
            x, y, out_file='./other/feature_importance.png')
        print('Finished getting feature importance')

    if (create_hists):
        shutil.rmtree('./hists/', ignore_errors=True)
        os.mkdir('./hists/')
        os.mkdir('./hists/complete_hists/')
        for s in stations:
            print('Creating hists for station {}'.format(s))
            data = pd.read_csv('data/data_{}.csv'.format(s), delimiter=';')
            data = data.drop(fieldsToDrop, axis=1)

            for c in data.columns:
                data[c] = data[c].apply(
                    shift_invalid_values, args=(abs(data[c].mean()), c))

            # histograms for all features in one figure
            data.hist(figsize=(20, 10), bins=20)
            plt.savefig('hists/complete_hists/{}.png'.format(s))
            plt.close()

            os.mkdir('hists/{}/'.format(s))

            # histograms for separate features
            for c in data.columns:
                ax = data.hist(c, figsize=(10, 8), bins=100)
                plt.savefig('hists/{}/{}.png'.format(s, c))
                plt.close()

    if (create_corr):
        create_correlation_matrix(
            complete_station_data.drop(fieldsToDrop, axis=1),
            'correlation_matrix')

    if (create_features):
        data = pd.read_csv('data/data_11816.csv', delimiter=';')
        data = data.drop(fieldsToDrop, axis=1)
        plot_features(data)

    if (stable_weather):
        data = pd.read_csv('data/data_11816.csv', delimiter=';')
        data_len = data.shape[0]

        scores = []
        total_stable_count = 0
        period = 24
        max_samples = 1000
        all_stable_positions = []
        diffs = []
        reals = []

        # 0 - stable enough
        # 1 - not stable enough

        current_offset = 0

        for i in range(2 * period, data_len - period):
            diff = abs(data.loc[i, 'future_temp_shmu'] -
                       data.loc[i - period, 'future_temp_shmu'])
            diffs.append(diff)
            real = abs(data.loc[i, 'future_temp'] -
                       data.loc[i - period, 'future_temp'])
            reals.append(real)
            scores.append(0 if diff <= 1 else 1)

            if (len(scores) >= max_samples or i == data_len - period - 1):
                plt.figure(figsize=(12, 6))
                ax = plt.subplot(111)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                plt.plot(reals, 'b', label='total difference')
                plt.title(
                    'Stable weather analysis from sample {}'.format(
                        current_offset + 2 * period))
                plt.ylabel('Difference')
                plt.xlabel('Samples')

                adjusted_scores = scores.copy()
                for pos, v in enumerate(scores):
                    retain = True
                    # at least 6 stable hours one after another
                    # to mark position as really stable
                    for f in range(0, 6):
                        if (scores[pos - f] != 0):
                            retain = False
                    if (not retain):
                        adjusted_scores[pos] = 1

                x_axis = []
                y_axis = []

                for pos, v in enumerate(adjusted_scores):
                    if (v == 0):
                        total_stable_count += 1
                        x_axis.append(pos)
                        y_axis.append(v)
                        all_stable_positions.append(
                            pos + 2 * period + current_offset)

                ax.scatter(x_axis, y_axis, marker='.',
                           color='g', label='stable weather')
                plt.legend(loc=2)
                plt.savefig('stable/{}.png'.format(i))
                plt.close()

                current_offset += max_samples
                del scores[:]
                del diffs[:]
                del reals[:]

        print('total stable count', total_stable_count)
        pd.Series(all_stable_positions).to_csv(
            'stable/stable_times.csv', index=False)
