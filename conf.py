'''
stable
    active - wheather to evaluate predictions only at stable weather
    func   - function to determine if it is stable weather
    ide    - 'ignore different errors' do not use autocorrection if prev errors were different enough
    aos    - 'autocorrect_only_stable' add autocorrect data only if it is stable weather
'''

'''
scale - (False, 'min-max', 'standard')
'''

config = {
    'data': 'data_tmp/data_11816.csv',
    'stable': {
        'active': False,
        'func': 's1',
        'ide': True,
        'aos': True,
    },
    'models': [
        {
            'model': 'ols',
            'weight': False,
            'scale': False,
            'model_params': {
                'fit_intercept': True,
            },
            'window_length': 60,
            'window_period': 24,
        },
    ],
}

'''
config = {
    'data': 'data_tmp/data_11816.csv',
    'stable': {
        'active': True,
        'func': 's1',
        'ide': True,
        'aos': True,
    },
    'models': [
        {
            'model': 'ols',
            'weight': 0.9,
            'model_params': {
                'fit_intercept': True,
            },
            'window_length': 30,
            'window_period': 24,
            'autocorrect': 'error24',
        },
        {
            'model': 'ols',
            'model_params': {
                'fit_intercept': True,
            },
            'window_length': 60,
            'window_period': 24,
        },
    ],
}
'''
