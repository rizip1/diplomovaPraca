import numpy as np
import math


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


def error24_48_logvar(model_errors, pos=0, interval=0, window_length=0,
                      is_test_set=False):
    '''
    Return error variance for model errors at times t-24+i and t-48+i
    where i in [-2,-1,0,1,2]
    '''
    offset = interval * window_length
    spread = 2

    if (is_test_set):
        e1 = model_errors[-(spread + 24): -(-(spread + 1) + 24)]
        e2 = model_errors[-(spread + 48): -(-(spread + 1) + 48)]
        return math.log(np.var(e1 - e2), 2)

    autocorrect_col = np.array([])
    for i in range(offset + 24, 24, -interval):
        e1 = model_errors[-(i + spread): -(i - (spread + 1))]
        e2 = model_errors[-(i + spread + 24): -(i - (spread + 1) + 24)]
        autocorrect_col = np.append(
            autocorrect_col, math.log(np.var(e1 - e2), 2))

    return autocorrect_col


def can_use_autocorrect24(model_errors, interval, window_len):
    period = 24 * 1
    return len(model_errors) > (interval * window_len) + period


def can_use_autocorrect24_48(model_errors, interval, window_len):
    period = 24 * 2
    return len(model_errors) > (interval * window_len) + period


def can_use_autocorrect24_48_logvar(model_errors, interval, window_len):
    period = 24 * 2
    return len(model_errors) > (interval * window_len) + period + 2


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
    'error24_48_logvar': {
        'func': error24_48_logvar,
        'can_use_auto': can_use_autocorrect24_48_logvar,
        'merge': merge24,
    },
}


def get_autocorrect_conf(key):
    if (not key):
        return None
    return autocorrect_map[key]
