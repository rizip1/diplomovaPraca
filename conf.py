config = {
    'data': 'data_tmp/data_11816.csv',
    'stable': {
        'active': True,
        'func': 's1',
        'ise': True,
    },
    'models': [
        {
            'model': 'svr',
            'model_params': {
                'fit_intercept': True,
            },
            'window_length': 60,
            'window_period': 24,
            'autocorrect': 'error24_48',
        },
    ],
}
