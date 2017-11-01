import matplotlib.pyplot as plt
import shutil
import csv
import os
import numpy as np
import pandas as pd
import argparse
import re
from datetime import datetime, timedelta

from utils import parse_month, parse_hour
from utils import save_hour_value, plot_hour_results

from db import save_data_for_station, get_stations
from sklearn.ensemble import ExtraTreesRegressor

pd.options.mode.chained_assignment = None


def get_parser():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', action='store_true', default=False,
                        dest='data',
                        help='Will recreate data files')
    parser.add_argument('--data-missing', action='store_true', default=False,
                        dest='data_missing',
                        help='Will check missing files')
    parser.add_argument('--invalid', action='store_true', default=False,
                        dest='invalid',
                        help='Will recreate invalid data')
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
    parser.add_argument('--compare-improvements', action='store_true',
                        default=False, dest='compare_improvements',
                        help='Will compare improvements')
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
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('{}/{}.png'.format('other', out_file))


def save_invalid_data_to_plots(folder, colors, invalid_rows_all):
    shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)
    label_mapping = {
        'current_pressure': 'Pressure',
        'current_wind_speed': 'Wind speed',
        'current_wind_direction': 'Wind direction',
        'current_rainfall_last_hour': 'Rainfall',
        'current_temp': 'Temperature',
        'current_humidity': 'Humidity',
        'future_temp_shmu': 'SHMU temperature',
    }
    label_order = ['current_pressure', 'current_wind_speed',
                   'current_wind_direction', 'current_rainfall_last_hour',
                   'current_temp', 'current_humidity', 'future_temp_shmu']
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
        plt.title('Missing data for station {}'.format(station))
        plt.ylabel('Features')
        plt.xlabel('Samples')
        plt.yticks([f for f in range(1, 8)])
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
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(20, 10))
    plt.title('Feature importances')
    plt.bar(range(x.shape[1]), importances[indices],
            color='r', align='center')
    plt.xticks(range(x.shape[1]), np.array(x.columns)[indices],
               rotation='horizontal')
    plt.xlim([-1, x.shape[1]])
    plt.savefig(out_file)
    plt.close()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    create_data = args.data
    create_invalid = args.invalid
    create_important = args.important
    create_hists = args.hists
    create_corr = args.corr
    create_features = args.features
    create_missing_data = args.data_missing
    create_shmu_errors = args.shmu_errors
    stable_weather = args.stable_weather
    compare_improvements = args.compare_improvements
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

                        match = re.search(
                            r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
                            new_val_date)
                        hour = int(match.group(1))
                        new_ref_date = None
                        match = re.search(
                            r'^([0-9]{4}-[0-9]{2}-[0-9]{2}) [0-9]{2}:[0-9]{2}:[0-9]{2}$',
                            new_val_date)
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
                                if (prev_r[c] == missing_v or next_r[c] == missing_v):
                                    prev_r[c] = missing_v
                                    next_r[c] = missing_v
                                # take average of t-24 and t+24
                                if (c != 'validity_date' and c != 'reference_date'):
                                    new_record[c] = (prev_r[c] + next_r[c]) / 2
                        else:
                            # just add row of all values set to -999
                            for c in columns:
                                if (c != 'validity_date' and c != 'reference_date'):
                                    new_record[c] = -999
                        rows.append(new_record.values)

            # last record that we have
            rows.append(s_data.loc[s_data.shape[0] - 1].values)

            new_data = pd.DataFrame(np.array(rows), columns=columns)
            new_data.to_csv('new_data/data_{}.csv'.format(s),
                            index=False, sep=';')

    if (create_missing_data):
        results = {}
        for s in stations:
            print('Processing station {}...'.format(s))
            total_missing_data = 0
            date_format = '%Y-%m-%d %H:%M:%S'
            s_data = pd.read_csv(
                'data/data_{}.csv'.format(s), delimiter=';')
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

    complete_station = 11894  # however no records about rainfall
    complete_station_data = pd.read_csv(
        'data/data_{}.csv'.format(complete_station), delimiter=';')
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
            data = pd.read_csv('data_tmp/data_{}.csv'.format(s), delimiter=';')
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
        save_invalid_data_to_plots(folder='./missing_data/', colors=colors,
                                   invalid_rows_all=invalid_rows_all)

    # Get most important features based on RandomForest training
    if (create_important):
        shutil.rmtree('./other/', ignore_errors=True)
        os.mkdir('./other/')
        print('Getting feature importance ...')

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

    if (compare_improvements):
        seasons = ['spring', 'summer', 'autumn', 'winter']
        x_m = [(i + 1) for i in range(12)]
        x_a = [(i + 13) for i in range(12)]

        for index, s in enumerate(seasons):
            base = pd.read_csv('improvement/compare/{}_base.csv'.format(s))
            alt = pd.read_csv('improvement/compare/{}_alt.csv'.format(s))

            plt.figure(figsize=(12, 6))
            plt.plot(x_m, base.morning, 'b', label='original')
            plt.plot(x_m, alt.morning, 'g', label='alternative')
            plt.title('Morning {}'.format(s))
            plt.ylabel('Improvement')
            plt.xlabel('Hour')
            plt.xticks(x_m)
            plt.grid()
            plt.legend(bbox_to_anchor=(0.97, 1.015), loc=2)
            plt.savefig('improvement/compared/morning_{}.png'.format(s))
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.plot(x_a, base.afternoon, 'b', label='original')
            plt.plot(x_a, alt.afternoon, 'g', label='alternative')
            plt.title('Afternoon {}'.format(s))
            plt.ylabel('Improvement')
            plt.xlabel('Hour')
            plt.xticks(x_a)
            plt.grid()
            plt.legend(bbox_to_anchor=(0.97, 1.015), loc=2)
            plt.savefig('improvement/compared/afternoon_{}.png'.format(s))
            plt.close()

    if (create_shmu_errors):
        data = pd.read_csv('data/data_11816.csv', delimiter=';')
        shmu_errors = [[] for i in range(24)]
        seasonal_shmu_errors = {
            'spring': [[] for i in range(24)],
            'summer': [[] for i in range(24)],
            'autumn': [[] for i in range(24)],
            'winter': [[] for i in range(24)],
        }

        data_len = data.shape[0]
        for i in range(data_len):
            val_date = data.validity_date[i]
            val_date_hour = parse_hour(val_date)
            val_date_month = parse_month(val_date)

            error = abs(
                data.loc[i, 'future_temp'] - data.loc[i, 'future_temp_shmu'])

            save_hour_value(shmu_errors, seasonal_shmu_errors,
                            error, val_date_hour, val_date_month)

        plot_hour_results(shmu_errors, seasonal_shmu_errors,
                          'SHMU errors', 'shmu_errors')
    if (stable_weather):
        data = pd.read_csv('data/data_11816.csv', delimiter=';')
        data_len = data.shape[0]

        scores = []
        window_pos = [-2, -1, 0, 1, 2]
        window_weights = [0.15, 0.6, 1, 0.6, 0.15]
        window_sum = np.sum(window_weights)
        total_stable_count = 0
        period = 24
        max_samples = 1000
        all_stable_positions = []

        # 0 - stable enough
        # 1 - not stable enough

        current_offset = 0

        for i in range(2 * period, data_len - period):
            score = 0
            for p in window_pos:
                weight = window_weights[p]
                score += weight * abs(data.loc[i + p, 'current_temp'] -
                                      data.loc[i + p - period, 'current_temp'])
            scores.append(score / window_sum)

            if (len(scores) >= max_samples or i == data_len - period - 1):
                plt.figure(figsize=(12, 6))
                ax = plt.subplot(111)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                plt.plot(scores, 'b', label='total difference')
                plt.title(
                    'Stable weather analysis from sample {}'.format(
                        current_offset + 2 * period))
                plt.ylabel('Difference')
                plt.xlabel('Samples')

                for pos, v in enumerate(scores):
                    scores[pos] = (v > 1) and 1 or 0

                for pos, v in enumerate(scores):
                    retain = True
                    # at least 4 stable hours one after another
                    # to mark position as really stable
                    for f in range(1, 5):
                        if (pos + f < len(scores)):
                            if (scores[pos + f] != 0):
                                retain = False
                    if (not retain):
                        scores[pos] = 1

                x_axis = []
                y_axis = []

                for pos, v in enumerate(scores):
                    if (v == 0):
                        total_stable_count += 1
                        x_axis.append(pos)
                        y_axis.append(v)
                        all_stable_positions.append(
                            pos + 2 * period + current_offset)

                ax.scatter(x_axis, y_axis, marker='.',
                           color='g', label='stable weather')
                plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
                plt.savefig('stable/{}.png'.format(i))
                plt.close()

                current_offset += max_samples
                del scores[:]

        print('total stable count', total_stable_count)
        pd.Series(all_stable_positions).to_csv(
            'stable/stable_times.csv', index=False)
