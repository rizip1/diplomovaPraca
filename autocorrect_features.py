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


def get_train_mean_1(model_errors, i):
    mean_offset = 12
    return np.mean(model_errors[-i - mean_offset:-i + 1])


def get_test_mean_1(model_errors):
    mean_offset = 12
    return np.mean(model_errors[-24 - mean_offset:-24 + 1])


def get_train_mean_2(model_errors, i):
    mean_offset = 12
    return np.mean(model_errors[-i:-i + mean_offset + 1])


def get_test_mean_2(model_errors):
    mean_offset = 12
    return np.mean(model_errors[-24:-24 + mean_offset + 1])


def get_train_mean_3(model_errors, i):
    mean_offset = 6
    return np.mean(model_errors[-i - mean_offset:-i + mean_offset + 1])


def get_test_mean_3(model_errors):
    mean_offset = 6
    return np.mean(model_errors[-24 - mean_offset:-24 + mean_offset + 1])


def errorMean24Common(trainSetFunc, testSetFunc, model_errors, pos=0,
                      interval=0, window_length=0, is_test_set=False):
    offset = interval * window_length

    if (is_test_set):
        mean = testSetFunc(model_errors)
        return np.array([model_errors[-24], mean])

    start = (offset + 24)
    means = []
    for i in range(start, 24, -24):
        mean = trainSetFunc(model_errors, i)
        means.append(mean)
    err24 = model_errors[-(offset + 24): -24: interval]
    return np.transpose(np.vstack((err24, np.array(means))))


def error24Mean1(model_errors, pos=0, interval=0, window_length=0,
                 is_test_set=False):
    '''
    both err24 and mean (-36, -24)
    '''
    return errorMean24Common(get_train_mean_1, get_test_mean_1, **locals())


def error24Mean2(model_errors, pos=0, interval=0, window_length=0,
                 is_test_set=False):
    '''
    both err24 and mean (-24, -12)
    '''
    return errorMean24Common(get_train_mean_2, get_test_mean_2, **locals())


def error24Mean3(model_errors, pos=0, interval=0, window_length=0,
                 is_test_set=False):
    '''
    both err24 and mean (-30, -18)
    '''
    return errorMean24Common(get_train_mean_3, get_test_mean_3, **locals())


def error24_48(model_errors, pos=0, interval=0, window_length=0,
               is_test_set=False):
    '''
    For each row in design matrix return the model error from 24
    hours before the time of making the prediction. For test
    vector return single value.
    '''
    offset = interval * window_length

    if (is_test_set):
        return np.array([model_errors[-24], model_errors[-48]])

    r1 = model_errors[-(offset + 24): -24: interval]
    r2 = model_errors[-(offset + 48): -48: interval]
    return np.transpose(np.vstack((r1, r2)))


def can_use_autocorrect24(model_errors, interval, window_len):
    period = 24 * 1
    return len(model_errors) > (interval * window_len) + period


def can_use_autocorrect24_48(model_errors, interval, window_len):
    period = 24 * 2
    return len(model_errors) > (interval * window_len) + period


def merge24(x_train, x_test, x_train_auto, x_test_auto, window_length):
    x_train_new = np.hstack((x_train, x_train_auto.reshape(window_length, 1)))
    x_test_new = np.hstack((x_test, x_test_auto))
    return (x_train_new, x_test_new)


def merge24_48(x_train, x_test, x_train_auto, x_test_auto, window_length):
    x_train_new = np.hstack((x_train, x_train_auto))
    x_test_new = np.hstack((x_test, x_test_auto))
    return (x_train_new, x_test_new)


# func - function to get autocorrect data
# can_use_auto - function to check if enough prediction to use autocorrect
# merge - function to merge autocorrect data with current data
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
    'error24Mean1': {
        'func': error24Mean1,
        'can_use_auto': can_use_autocorrect24_48,
        'merge': merge24_48,
    },
    'error24Mean2': {
        'func': error24Mean2,
        'can_use_auto': can_use_autocorrect24_48,
        'merge': merge24_48,
    },
    'error24Mean3': {
        'func': error24Mean3,
        'can_use_auto': can_use_autocorrect24_48,
        'merge': merge24_48,
    }
}


def get_autocorrect_conf(key):
    if (not key):
        return None
    return autocorrect_map[key]
