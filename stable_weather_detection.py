import numpy as np
import re

'''
No need to handle 'out of bound index' for any reasonable window length
'''


def handle_missing(v1, v2):
    return v1 != -999 and v2 != -999


def is_error_diff_enough(model_errors):
    threshold = 0.5
    if (len(model_errors) < 48):
        return True
    max_error = max(abs(model_errors[-24]), abs(model_errors[-48])) + 1
    error_diff = abs(model_errors[-24] - model_errors[-48]) + 1
    return (error_diff / max_error) < threshold


def s1(data, pos, offset1=24, offset2=48):
    '''
    Distances between temperatures in t-24+i and t-48+i must not be greater
    than THRESHOLD for i=[-2,-1,0,1,2]
    '''
    hours = [-2, -1, 0, 1, 2]
    threshold = 1

    for h in hours:
        v1 = data.loc[pos - offset1 + h, 'current_temp']
        v2 = data.loc[pos - offset2 + h, 'current_temp']
        handle_missing(v1, v2)
        if (abs(v1 - v2) > threshold):
            return False
    return True


def s2(data, pos, offset1=24, offset2=48):
    '''
    Gaussian weighted distances between temperatures in t-24+i and t-48+i
    must not be greater than THRESHOLD
    '''
    hours = [-2, -1, 0, 1, 2]
    gauss_weights = [0.15, 0.6, 1, 0.6, 0.15]
    window_sum = np.sum(gauss_weights)
    threshold = 1
    dist = 0

    for i, h in enumerate(hours):
        v1 = data.loc[pos - offset1 + h, 'current_temp'] * gauss_weights[i]
        v2 = data.loc[pos - offset2 + h, 'current_temp'] * gauss_weights[i]
        handle_missing(v1, v2)
        dist += abs(v1 - v2)

    dist /= window_sum
    if (dist > threshold):
        return False
    return True


def s3(data, pos, offset1=0, offset2=24):
    '''
    Distances between temperatures in ref_time+i and ref_time-24+i must not
    be greater than THRESHOLD for i=[-2,-1,0,1,2]
    '''
    hours = [-4, -3, -2, -1, 0]
    threshold = 1

    val_date = data.loc[pos, 'validity_date']
    m = re.search(
        r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
        val_date)
    hour = int(m.group(1))
    offset = hour
    if (hour > 12):
        offset -= 12
    if (hour == 0):
        offset = 12
    offset -= 1

    o_pos = pos - offset

    for h in hours:
        v1 = data.loc[o_pos - offset1 + h, 'current_temp']
        v2 = data.loc[o_pos - offset2 + h, 'current_temp']
        handle_missing(v1, v2)
        if (abs(v1 - v2) > threshold):
            return False
    return True


def s4(data, pos):
    '''
    mark as stable also if weather conditions from last day
    match well with weather conditions measured three days ago.
    consider prediction time
    '''
    return s1(data, pos, offset1=24, offset2=72) or s1(data, pos)


def s5(data, pos):
    '''
    mark as stable also if weather conditions from last day
    match well with weather conditions measured three days ago.
    consider prediction time
    '''
    return s2(data, pos, offset1=24, offset2=72) or s2(data, pos)


def s6(data, pos):
    '''
    mark as stable also if weather conditions from time before making
    the prediction match well with weather conditions measured
    two days ago (from time before making the prediction)
    '''
    return s3(data, pos, offset1=0, offset2=48) or s3(data, pos)


def s7(data, pos, offset=24):
    '''
    Distances between predicted temperatures in t+i and t-24+i must not
    be greater than THRESHOLD for i=[-2,-1,0]
    '''
    hours = [-5, -4, -3, -2, -1, 0]
    threshold = 1

    for h in hours:
        v1 = data.loc[pos + h, 'future_temp_shmu']
        v2 = data.loc[pos - offset + h, 'future_temp_shmu']
        handle_missing(v1, v2)
        if (abs(v1 - v2) > threshold):
            return False
    return True


def union1(data, pos):
    return s1(data, pos) or s2(data, pos) or s3(data, pos)


def union2(data, pos):
    return s4(data, pos) or s5(data, pos) or s6(data, pos)


stable_functions = {
    's1': s1,
    's2': s2,
    's3': s3,
    's4': s4,
    's5': s5,
    's6': s6,
    's7': s7,
    'union1': union1,
    'union2': union2,
}


def get_stable_func(key):
    if (not key):
        return None
    return stable_functions[key]
