import re
import math
import numpy as np


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


def shmu_prediction_time_error(data, lags=1, lag_by=1, exp=0):
    '''
    For `lag` = 1 add shmu error in time when prediction
    was created.
    For `lag` = n add `n` off stese values, every lagged by
    one more hour

    `lags` = 1 means no lag, use only last prediction error
    `lag_by` if `lags` > 1, what should be lag distance
    '''
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


def temperature_prediction_time_var(data, samples_count):
    if (samples_count == 0):
        return data
    data_size = data.shape[0]

    # add 1 because current temp is calculated from future_temp
    # TODO rework this later
    new_start = 12 + samples_count + 1

    new_col = 'temperature_var_{}'.format(samples_count)
    data[new_col] = 0  # will set all rows to zero

    for j in range(new_start, data_size):
        ref_date = data.loc[j, 'validity_date']
        m = re.search(
            r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
            ref_date)
        hour = int(m.group(1))
        if (hour > 12):
            hour -= 12

        temp_start = j - hour - (samples_count + 1)
        temp_end = j - hour - 1
        data.loc[j, new_col] = np.var(
            data.loc[temp_start: temp_end, 'future_temp'])

    # get rid of rows for which we do not have data
    return data.iloc[new_start:data_size, :].reset_index(drop=True)


def shmu_error_prediction_time_var(data, samples_count):
    if (samples_count == 0):
        return data
    data_size = data.shape[0]

    new_start = 12 + samples_count

    new_col = 'shmu_error_var_{}'.format(samples_count)
    data[new_col] = 0  # will set all rows to zero

    for j in range(new_start, data_size):
        ref_date = data.loc[j, 'validity_date']
        m = re.search(
            r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
            ref_date)
        hour = int(m.group(1))
        if (hour > 12):
            hour -= 12

        var_values = []
        for i in range(samples_count):
            sh_error = data.loc[j - hour - i, 'future_temp'] - \
                data.loc[j - hour - i, 'future_temp_shmu']
            var_values.append(sh_error)

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


def non_linear_transform(data):
    # TODO make CLI swiches for this
    for i in range(data.shape[0]):
        # data.loc[i, 'test'] = data.loc[i, 'current_temp'] ** 2
        '''
        if (data.loc[i, 'current_temp'] == 0):
            data.loc[i, 'test'] = 0
        else:
            data.loc[i, 'test'] = math.log(
                data.loc[i, 'current_temp'] ** 2, 10)
        '''
        # data.loc[i, 'test'] = cubic_root(data.loc[i, 'current_temp'])

        '''
        data.loc[i, 'test2'] = cubic_root(
            data.loc[i, 'future_temp_shmu'] - data.loc[i, 'current_temp'])
        '''

        '''
        data.loc[i, 'test2'] = math.log(
            abs(data.loc[i, 'future_temp_shmu'] - data.loc[i, 'current_temp']),
            2)
        '''

        '''
        data.loc[i, 'test2'] = math.pow(
            data.loc[i, 'future_temp_shmu'] - data.loc[i, 'current_temp'], 2)
        '''

        '''
        data.loc[i, 'test2'] = math.pow(
            data.loc[i, 'future_temp_shmu'] - data.loc[i, 'current_temp'], 3)
        '''

        '''
        data.loc[i, 'test2'] = cubic_root(
            data.loc[i, 'future_temp_shmu'] * data.loc[i, 'current_temp'])
        '''

        '''
        data.loc[i, 'test2'] = data.loc[
            i, 'future_temp_shmu'] * data.loc[i, 'current_temp']
        '''
        '''
        data.loc[i, 'test2'] = math.log(max(0.001, abs(data.loc[
            i, 'future_temp_shmu'] * data.loc[i, 'current_temp'])), 2)
        '''
        '''
        data.loc[i, 'test2'] = data.loc[
            i, 'future_temp_shmu'] - data.loc[i, 'current_temp']
        '''

    return data.reset_index(drop=True)
