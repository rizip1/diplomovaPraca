config = {
    'data': 'data_tmp/data_11816.csv',
    'stable_func': 'union',
    'models': [
        {
            'model': 'ols',
            'model_params': {
                'fit_intercept': True,
            },
            'weight': 0.90,
            'window_length': 10,
            'window_period': 24,
            'autocorrect': 'err',
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
