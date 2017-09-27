import argparse


def get_predict_parser():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--weight', action='store', dest='weight_coef',
                        help='''Weight coefficient. If none supplied, no
weights will be used.''', type=float)
    parser.add_argument('--file', action='store', dest='data_file',
                        required=True,
                        help='''Path to data file that will be loaded.''')
    parser.add_argument('--mode', action='store', dest='mode',
                        default='window',
                        choices=['window', 'extended-window', 'train-set'],
                        help="""Mode to use for predictions:\n
window = use sliding window
extended-window = use window that grows over time
train-set = learn from fixed length train set\n
Default length for window, extended-window and train-set is 60.
To override it set '--length' option.""")
    parser.add_argument('--length', action='store', dest='length',
                        default=60,
                        help='Length of window, extended-window or train-set.')
    parser.add_argument('--lags', action='store', dest='lags',
                        default=0,
                        help='Length of window, extended-window or train-set.')
    parser.add_argument('--no-intercept', action='store_true', default=False,
                        dest='no_intercept',
                        help='If set will not use bias term.')
    parser.add_argument('--model', action='store', dest='model',
                        default='reg',
                        choices=['reg', 'svr', 'rf', 'kn', 'nn',
                                 'ens', 'ens-linear', 'ens-ens'],
                        help="Model to use for predictions:\n")
    parser.add_argument('--shmu-error', action='store',
                        dest='shmu_error',
                        default=0,
                        help='Will use shmu error from "arg" hours before \n')
    parser.add_argument('--shmu-error-p-time', action='store',
                        dest='shmu_error_p_time',
                        default='0:1:0',
                        help='''Will use shmu error from time when prediction
was made. First agr specifies lags count. For no lag set it equal to 1.
Second arg specifies lag distance in hours. Default is 1.
Third arg specifies exponent func, 0 means no exponent func. \n''')
    parser.add_argument('--feature-p-time', action='store',
                        dest='feature_p_time',
                        help='''Except input in format lag_count:lag_by:feature_name.
The supplied feature will be lagged by count hours from prediction time,
including each lag.\n''')
    parser.add_argument('--feature', action='store',
                        dest='feature',
                        help='''Except input in format lag_count:lag_by:feature_name.
The supplied feature will be lagged by count hours, including each lag.\n''')
    parser.add_argument('--moments', action='store',
                        default=0,
                        dest='moments',
                        help='''Add temperature moment. Using values from
time when prediction was made and 12 hours before. Possible options
are 'mean', 'var', 'skew', 'kur'. To combination use format
'moment1-moment2 ... '\n''')
    parser.add_argument('--min-max', action='store',
                        default=0,
                        dest='min_max',
                        help='''Add min/max value for temperature using values from
time when prediction was made and 12 hours before. Possible options
are 'min', 'max', 'min-max'.\n''')
    parser.add_argument('--shmu-error-moment', action='store',
                        default=0,
                        dest='shmu_error_moment',
                        help='''Add shmu error moment from
time when prediction was made and arg hours before. Options are
'mean', 'var', 'mean-var'.\n''')
    parser.add_argument('--diff', action='store_true', dest='diff',
                        default=False,
                        help='Perform one step difference')
    parser.add_argument('--step', action='store', dest='step',
                        default=12,
                        help='Hour interval between learning examples')
    parser.add_argument('--norm', action='store_true', dest='norm',
                        default=False,
                        help='Normalize with mean and std')
    parser.add_argument('--avg', action='store_true', dest='average_models',
                        default=False,
                        help='Average models')
    parser.add_argument('--autocorrect', action='store',
                        dest='autocorrect',
                        choices=['err', 'err-var', 'err-sum-var'],
                        default=False,
                        help='Use autocorrection')
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        default=False,
                        help='Verbose output')
    parser.add_argument('--use-cache', action='store_true', dest='use_cache',
                        default=False,
                        help='Used cached data')
    parser.add_argument('--skip-predictions', action='store',
                        dest='skip_predictions', default=0,
                        help='''Number of predictions to do not count to final
model score. Used to compare models with different window lengths.\n''')
    return parser
