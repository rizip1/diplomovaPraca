'''
stable
    active - wheather to evaluate predictions only at stable weather
    func   - function to determine if it is stable weather
    ide    - 'ignore different errors' do not use autocorrection if prev errors were different enough
    aos    - 'autocorrect_only_stable' add autocorrect data only if it is stable weather

models:
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
                'feature-lagged-p-time': {
                    'lags': 1,
                    'lag_by': 12,
                    'name': 'current_temp',
                },
                'feature-lagged': {
                    'lags': 1,
                    'lag_by': 12,
                    'name': 'current_temp',
                },
            },
        },
        {
            'model': 'ols',
            'weight': False,
            'scale': False,
            'model_params': {
                'fit_intercept': True,
            },
            'window_length': 60,
            'window_period': 24,
            'features': {
                'moments': 'var',
            },
        },
    ],
}