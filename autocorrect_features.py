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


def error24_48(model_errors, pos=0, interval=0, window_length=0,
               is_test_set=False):
    '''
    For each row in design matrix return the model error from 24
    hours before the time of making the prediction. For test
    vector return single value.
    '''
    offset = interval * window_length

    if (is_test_set):
        return np.array([model_errors[-48], model_errors[-24]])

    r1 = model_errors[-(offset + 24): -24: interval]
    r2 = model_errors[-(offset + 48): -48: interval]
    return np.transpose(np.vstack((r1, r2)))


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


def can_use_autocorrect24(model_errors, interval, window_len):
    period = 24 * 2
    return len(model_errors) > (interval * window_len) + period


def can_use_autocorrect24_48(model_errors, interval, window_len):
    period = 24 * 3
    return len(model_errors) > (interval * window_len) + period


def merge24(x_train, x_test, x_train_auto, x_test_auto, window_length):
    x_train_new = np.hstack((x_train, x_train_auto.reshape(window_length, 1)))
    x_test_new = np.hstack((x_test, x_test_auto))
    return (x_train_new, x_test_new)


def merge24_48(x_train, x_test, x_train_auto, x_test_auto, window_length):
    x_train_new = np.hstack((x_train, x_train_auto))
    x_test_new = np.hstack((x_test, x_test_auto))
    return (x_train_new, x_test_new)

'''
# func - function to get autocorrect data
# can_use_auto - function to check if enough prediction to use autocorrect
# merge - function to merge autocorrect data with current data
'''
autocorrect_map = {
    'error24': {
        'func': error24,
        'can_use_auto': can_use_autocorrect24,
        'merge': merge24,
    },
    'error24_48': {
        'func': error24_48,
        'can_use_auto': can_use_autocorrect24_48,
        'merge': merge24_48,
    },
}


def get_autocorrect_conf(key):
    if (not key):
        return None
    return autocorrect_map[key]
