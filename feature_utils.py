import re
import math
import numpy as np
import pandas as pd

'''
TODO refactor/reuse code
'''


def feature_lagged_by_hours_p_time(data, feature, lags, lag_by=12):
    '''
    Add feature lagged by hours from prediction time
    '''

    if (lags == 0 or feature is None or lag_by == 0):
        return data

    data_size = data.shape[0]
    new_start = lag_by * lags

    for i in range(lags):
        # add column of zeros for each lag
        new_col = '{}_lag_p_time_{}_{}'.format(feature, i + 1, lag_by)
        data[new_col] = 0  # will set all rows to zero

        start = (i + 1) * lag_by + lag_by
        for j in range(start, data.shape[0]):
            ref_date = data.loc[j, 'validity_date']
            m = re.search(
                r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
                ref_date)
            hour = int(m.group(1))
            if (hour == 0):
                hour = 24
            if (hour > 12):
                hour -= 12

            # this is because otherwise we would get one time step before
            # required record, ok for shmu temp error
            hour += 1

            data.loc[j, new_col] = data.loc[
                j - hour - ((i + 1) * lag_by), feature]

    # get rid of values that dont have lagged values
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def feature_lagged_by_hours(data, feature, lags, lag_by=12):
    if (feature is None or lags == 0 or lag_by == 0):
        return data
    data_size = data.shape[0]
    new_start = lags * lag_by
    for i in range(lags):
        # add column of zeros for each lag
        new_col = '{}_lag_{}_{}'.format(feature, i + 1, lag_by)
        data[new_col] = 0  # will set all rows to zero

        start = i * lag_by + lag_by
        for j in range(start, data.shape[0]):
            data.loc[j, new_col] = data.loc[j - start, feature]

    # get rid of values that dont have lagged values
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def add_shmu_error(data, before):
    if (before == 0):
        return data
    '''
    Add shmu error made `before` hours before current time
    '''
    data_size = data.shape[0]
    new_start = before
    new_col = 'shmu_error_{}'.format(before)
    data[new_col] = 0  # will set all rows to zero

    for j in range(new_start, data.shape[0]):
        data.loc[j, new_col] = data.loc[
            j - before, 'future_temp_shmu'] - data.loc[
            j - before, 'future_temp']

    # get rid of rows for which we do not have data
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def shmu_prediction_time_error(data, lags=1, lag_by=1, exp=0):
    '''
    For `lag` = 1 add shmu error in time when prediction
    was created.
    For `lag` = n add `n` off stese values, every lagged by
    one more hour

    `lags` = 1 means no lag, use only last prediction error
    `lag_by` if `lags` > 1, what should be lag distance
    '''
    if (lags == 0):
        return data

    data_size = data.shape[0]

    # remove rows for which we can not get lagged value
    # 12 is used to be safe within all hours
    predictions_ahead = 12
    new_start = predictions_ahead + (lag_by * lags)

    for i in range(lags):
        # add column of zeros for each lag
        new_col = 'shmu_pred_err_{}_{}'.format(i + 1, lag_by)
        data[new_col] = 0  # will set all rows to zero

        start = (i * lag_by) + predictions_ahead
        for j in range(start, data.shape[0]):
            ref_date = data.loc[j, 'validity_date']
            m = re.search(
                r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
                ref_date)
            hour = int(m.group(1))
            if (hour == 0):
                hour = 24
            if (hour > 12):
                hour -= 12

            data.loc[j, new_col] = data.loc[
                j - hour - ((i * lag_by)), 'future_temp_shmu'] - data.loc[
                j - hour - ((i * lag_by)), 'future_temp']

            if (exp != 0):
                value = data.loc[j, new_col]
                tmp = math.pow(exp, abs(value))
                if (value < 0):
                    tmp = -tmp
                data.loc[j, new_col] = tmp

    # get rid of rows for which we do not have data
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def add_morning_and_afternoon_temp(data, perform=False):
    '''
    Add temperature measured at 6pm for predictions between
    1-12am and at 6am for predictions between 13-00pm.
    '''
    if (not perform):
        return data
    data_size = data.shape[0]
    new_start = 12
    new_col = 'afternoon-morning_temp'
    data[new_col] = 0  # will set all rows to zero

    for j in range(new_start, data.shape[0]):
        val_date = data.loc[j, 'validity_date']
        m = re.search(
            r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
            val_date)
        hour = int(m.group(1))
        if (hour == 0):
            hour = 24
        if (hour > 12):
            hour -= 12  # get one hour before the prediction was made
        data.loc[j, new_col] = data.loc[j - hour - 5, 'current_temp']

    # get rid of rows for which we do not have data
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def add_moments(data, moments):
    if (not moments):
        return data

    samples = 12
    splitted = moments.split('-')
    data_size = data.shape[0]
    new_start = samples
    options = ['mean', 'var', 'skew', 'kur']

    for moment in splitted:
        if (moment not in options):
            raise Exception(
                'Invalid option supplied for moments: {}'.format(moment))
        data[moment] = 0

        for j in range(new_start, data_size):
            ref_date = data.loc[j, 'validity_date']
            m = re.search(
                r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
                ref_date)
            hour = int(m.group(1))
            if (hour == 0):
                hour = 24
            if (hour > 12):
                hour -= 12

            temp_start = j - hour - samples + 1
            temp_end = j - hour + 2
            values = data.loc[temp_start: temp_end, 'current_temp']

            if (moment == 'mean'):
                data.loc[j, moment] = np.mean(values)
            elif (moment == 'var'):
                data.loc[j, moment] = np.var(values)
            elif (moment == 'skew'):
                data.loc[j, moment] = pd.Series(values).skew()
            elif (moment == 'kur'):
                data.loc[j, moment] = pd.Series(values).kurtosis()

    # get rid of rows for which we do not have data
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def add_min_max(data, min_max):
    if (not min_max):
        return data

    samples = 12
    splitted = min_max.split('-')
    data_size = data.shape[0]
    new_start = samples
    options = ['min', 'max']

    for option in splitted:
        if (option not in options):
            raise Exception(
                'Invalid option supplied for min-max: {}'.format(option))
        data[option] = 0

        for j in range(new_start, data_size):
            ref_date = data.loc[j, 'validity_date']
            m = re.search(
                r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
                ref_date)
            hour = int(m.group(1))
            if (hour == 0):
                hour = 24
            if (hour > 12):
                hour -= 12

            temp_start = j - hour - samples + 1
            temp_end = j - hour + 2
            values = data.loc[temp_start: temp_end, 'current_temp']

            if (option == 'min'):
                data.loc[j, option] = min(values)
            elif (option == 'max'):
                data.loc[j, option] = max(values)

    # get rid of rows for which we do not have data
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def shmu_error_prediction_time_moment(data, moments):
    if (not moments):
        return data
    data_size = data.shape[0]
    samples = 12
    new_start = samples + samples
    splitted = moments.split('-')
    options = ['mean', 'var']

    for option in splitted:
        if (option not in options):
            raise Exception(
                'Invalid option supplied for shmu error moments: {}'
                .format(option))
        data[option] = 0

        new_col = 'shmu_error_{}'.format(option)
        data[new_col] = 0  # will set all rows to zero

        for j in range(new_start, data_size):
            ref_date = data.loc[j, 'validity_date']
            m = re.search(
                r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
                ref_date)
            hour = int(m.group(1))
            if (hour == 0):
                hour = 24
            if (hour > 12):
                hour -= 12

            var_values = []
            for i in range(samples + 1):
                sh_error = data.loc[j - hour - i, 'future_temp'] - \
                    data.loc[j - hour - i, 'future_temp_shmu']
                var_values.append(sh_error)

            if (option == 'mean'):
                data.loc[j, new_col] = np.mean(var_values)
            elif (option == 'var'):
                data.loc[j, new_col] = np.var(var_values)

    # get rid of rows for which we do not have data
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def cubic_root(x):
    if x > 0:
        return math.pow(x, float(1) / 3)
    elif x < 0:
        return -math.pow(abs(x), float(1) / 3)
    else:
        return 0
