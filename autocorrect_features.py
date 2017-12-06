import numpy as np


def error24(model_errors, pos=0, interval=0, window_length=0,
            is_test_set=False):
    '''
    For each row in design matrix return the model error from 24
    hours before the time of making the prediction. For test
    vector return single value.
    '''
    offset = interval * window_length

    if (is_test_set):
        return model_errors[-24]

    return model_errors[-(offset + 24): -24: interval]


def get_autocorrect_err2(model_errors, pos=0, interval=0, window_length=0,
                         is_test_set=False):
    '''
    TODO
    '''
    autocorrect_col = np.array([])
    offset = interval * window_length

    if (is_test_set):
        pos_day_before = pos - 24 - offset
        pos_two_days_before = pos - 48 - offset
        e1 = model_errors[pos_two_days_before - 3: pos_two_days_before + 4]
        e2 = model_errors[pos_day_before - 3: pos_day_before + 4]

        total_diff = 0
        for i in range(len(e1)):
            total_diff += abs(e1[i] - e2[i])
        return 1 / max(total_diff, 0.01)

    for i in range(pos - 24 - (2 * offset), pos - 24 - offset, interval):
        e1 = model_errors[i - 3: i + 4]
        e2 = model_errors[i - 3 - 24: i + 4 - 24]

        total_diff = 0
        for i in range(len(e1)):
            total_diff += abs(e1[i] - e2[i])

        autocorrect_col = np.append(autocorrect_col, 1 / max(total_diff, 0.01))

    return autocorrect_col


autocorrect_map = {
    'error24': error24,
    'err2': get_autocorrect_err2,
}


def get_autocorrect_func(key):
    if (not key):
        return None
    return autocorrect_map[key]
