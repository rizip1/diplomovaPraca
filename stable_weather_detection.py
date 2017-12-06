import numpy as np
import re

'''
TODO look one more day ago
'''


def handle_missing(v1, v2):
    return v1 != -999 and v2 != -999


def s1(data, pos):
    '''
    Distances between temperatures in t-24+i and t-48+i must not be greater
    than THRESHOLD for i=[-2,-1,0,1,2]
    '''
    hours = [-2, -1, 0, 1, 2]
    threshold = 1

    for h in hours:
        b_24 = data.loc[pos - 24 + h, 'current_temp']
        b_48 = data.loc[pos - 48 + h, 'current_temp']
        handle_missing(b_24, b_48)
        if (abs(b_48 - b_24) > threshold):
            return False
    return True


def s2(data, pos):
    '''
    Gaussian weighted distances between temperatures in t-24+i and t-48+i
    must not be greater than THRESHOLD for i=[-2,-1,0,1,2]
    '''
    hours = [-2, -1, 0, 1, 2]
    gauss_weights = [0.15, 0.6, 1, 0.6, 0.15]
    window_sum = np.sum(gauss_weights)
    threshold = 1

    for h in hours:
        dist = 0
        for w in gauss_weights:
            b_24 = data.loc[pos - 24 + h, 'current_temp'] * w
            b_48 = data.loc[pos - 48 + h, 'current_temp'] * w
            handle_missing(b_24, b_48)
            dist += abs(b_48 - b_24)
        dist /= window_sum
        if (dist > threshold):
            return False
    return True


def s3(data, pos):
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
        b_24 = data.loc[o_pos - 24 + h, 'current_temp']
        b_48 = data.loc[o_pos - 48 + h, 'current_temp']
        handle_missing(b_24, b_48)
        if (abs(b_48 - b_24) > threshold):
            return False
    return True


def union(data, pos):
    return s1(data, pos) or s2(data, pos) or s3(data, pos)


stable_functions = {
    's1': s1,
    's2': s2,
    's3': s3,
    'union': union,
}


def get_stable_func(key):
    if (not key):
        return None
    return stable_functions[key]
