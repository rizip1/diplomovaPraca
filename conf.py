config = {
    'data': 'data_tmp/data_11816.csv',
    'stable': {
        'active': True,
        'func': 's2',
        'ise': True,
    },
    'models': [
        {
            'model': 'ols',
            'model_params': {
                'fit_intercept': True,
            },
            'weight': 0.93,
            'window_length': 10,
            'window_period': 24,
            'autocorrect': 'error24',
        },
        {
            'model': 'ols',
            'model_params': {
                'fit_intercept': True,
            },
            'weight': 0.97,
            'window_length': 60,
            'window_period': 24,
        },
    ],
}
