'''
stable
    active - wheather to evaluate predictions only at stable weather
    func   - function to determine if it is stable weather
    ide    - 'ignore different errors' do not use autocorrection if prev errors were different enough
    aos    - 'autocorrect_only_stable' add autocorrect data only if it is stable weather

models:
    scale - (False, 'min-max', 'standard')
'''

# use for switch off/on of stable weather

config = {
    'data': 'new_data/data_11816.csv',
    'stable': {
        'active': False,
        'func': 's1',
        'ide': False,  # ignore for now
        'aos': True,
    },
    'models': [
        {
            'model': 'ols',
            'weight': 0.97,
            'diff': False,
            'scale': False,
            'model_params': {
                'fit_intercept': True,
            },
            'window_length': 120,
            'window_period': 24,
            'skip': 0,
            'features': {
                'min-max': 'min-max',
                'feature-lagged': [{
                    'lags': 1,
                    'lag_by': 1,
                    'name': 'future_temp_shmu',
                }],
            },
        },
        {
            'model': 'ols',
            'weight': 0.97,
            'diff': False,
            'scale': False,
            'model_params': {
                'fit_intercept': True,
            },
            'window_length': 120,
            'window_period': 24,
            'skip': 0,
            'features': {
                'moments': 'mean',
                'feature-lagged': [{
                    'lags': 1,
                    'lag_by': 1,
                    'name': 'future_temp_shmu',
                }],
            },
        },
        {
            'model': 'ols',
            'weight': 0.97,
            'diff': False,
            'scale': False,
            'model_params': {
                'fit_intercept': True,
            },
            'window_length': 120,
            'window_period': 24,
            'skip': 0,
            'features': {
                'afternoon-morning': True,
                'feature-lagged': [{
                    'lags': 1,
                    'lag_by': 1,
                    'name': 'future_temp_shmu',
                }],
            },
        },
    ],
}

# Note: as implemented now, 'diff' can not be used with 'autocorrection'

# TEMPLATE
'''
    {
        'model': 'ols',
        'weight': False,
        'scale': False,
        'model_params': {
            'fit_intercept': True,
        },
        'window_length': 60,
        'window_period': 24,
        'autocorrect': 'error24',
        'diff': False,
        'skip': 0,
        'features': {
            'afternoon-morning': True,
            'moments': 'mean',
            'min-max': 'min-max',
            'shmu-error': 24,
            'shmu-error-moment': 'mean',
            'shmu-error-p-time': {
                'lags': 1,
                'lag_by': 1,
                'exp': 0,
            },
            'feature-lagged-p-time': [
                {
                    'lags': 3,
                    'lag_by': 1,
                    'name': 'future_temp_shmu',
                },
                {
                    'lags': 3,
                    'lag_by': 1,
                    'name': 'current_temp',
                },
            ],
            'feature-lagged': [{
                'lags': 1,
                'lag_by': 1,
                'name': 'future_temp_shmu',
            }],
        },
    },
'''
