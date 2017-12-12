import matplotlib.pyplot as plt
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson

from scipy.stats.mstats import normaltest
from scipy.stats import norm, kstest

from utils import color_print, parse_hour
from constants import ERRORS_PATH, ERRORS_AUTOCOR_PATH
from constants import ERRORS_ERRORS_PATH


def _plot_normality(errors):
    color_print('\nErrors normality tests')
    print('Null hypothesis states that the error distribution is normal')

    # based on D’Agostino and Pearson’s test that combines
    # skew and kurtosis to produce an omnibus test of normality.
    p1 = normaltest(errors).pvalue
    print('D’Agostino and Pearson’s p-value: {0:.4f}'.format(p1))

    p2 = kstest(errors, 'norm').pvalue
    print('Kolmogorov-Smirnov p-value: {0:.4f}'.format(p2))

    mu, std = norm.fit(errors)
    plt.figure(figsize=(12, 6))
    plt.hist(errors, bins=30, normed=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.savefig('{}/hist.png'.format(ERRORS_PATH))
    plt.close()

    qqplot(errors)
    plt.savefig('{}/qq.png'.format(ERRORS_PATH))
    plt.close()


def _plot_all_errors(errors, max_count=10000000):
    '''
    If 'max_count' not specified save all errors to one graph
    '''
    print('\nSaving error plots ...')
    for i in range(0, len(errors), max_count):
        end = min(i + max_count, len(errors))
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.plot(errors[i:end], 'k')
        plt.title('Errors {}-{}'.format(i, end))
        plt.ylabel('Error')
        plt.xlabel('Samples')
        plt.savefig('{}/{}-{}.png'.format(ERRORS_ERRORS_PATH, i, end))
        plt.close()


def _get_hour_errors(data):
    hour_errors = [[] for i in range(24)]

    for i in data.index:
        val_date = data.loc[i, 'validity_date']
        hour = parse_hour(val_date)

        error = data.loc[i, 'predicted'] - data.loc[i, 'future_temp']
        hour_errors[hour - 1].append(error)

    return hour_errors


def _save_autocorrelation(hour_errors):
    '''
    Durbin-Watson
    http://www.statisticshowto.com/durbin-watson-test-coefficient/
        2 is no autocorrelation.
        0 to <2 is positive autocorrelation (common in time series data).
        >2 to 4 is negative autocorrelation (less common in time series data).

        rule of thumb is that test statistic values in the range
        of 1.5 to 2.5 are relatively ok
    '''
    print('Saving errors autocorrelation stats ...')
    with open('{}/durbin_watson.txt'.format(ERRORS_PATH), 'w') as f:
        for i in range(24):
            dw = durbin_watson(hour_errors[i])
            f.write('{0}\t {1:.4f} \n'.format(i + 1, dw))
            plot_acf(hour_errors[i], lags=60, alpha=0.05)
            plt.savefig('{}/{}.png'.format(ERRORS_AUTOCOR_PATH,  i + 1))
            plt.close()


def save_errors(data):
    errors = data.predicted - data.future_temp
    _plot_normality(errors)
    _plot_all_errors(errors, max_count=500)

    hour_errors = _get_hour_errors(data)
    _save_autocorrelation(hour_errors)
