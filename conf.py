config = {
    'data': 'data_tmp/data_11816.csv',
    'models': [
        {
            'model': 'ols',
            'model_params': {
                'fit_intercept': True,
            },
            'weight': 0.97,
            'window_length': 60,
            'window_period': 12,
            'autocorrect': 'err',
        },
        {
            'model': 'ols',
            'model_params': {
                'fit_intercept': True,
            },
            'weight': 0.97,
            'window_length': 60,
            'window_period': 12,
        },
    ],
}
