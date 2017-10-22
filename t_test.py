import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats

from scipy.stats import ttest_ind


def mean_confidence_interval(data, confidence=0.95):
    # TODO add stack-overflow source
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

err1 = np.squeeze(pd.read_csv('other/compare_errors/errors_24_60.csv').values)
err2 = np.squeeze(pd.read_csv('other/compare_errors/24_60_a-m.csv').values)

samples_count = min(err1.shape[0], err2.shape[0])

err1_abs = np.absolute(err1[-samples_count:])
err2_abs = np.absolute(err2[-samples_count:])

print('Compared on {} samples'.format(samples_count))

# t-test
print('p-value for equal means:',
      ttest_ind(err1_abs, err2_abs, equal_var=False).pvalue)

# box and whisker plot
# notches represent the confidence interval (CI) around the median
plt.boxplot([err1_abs, err2_abs], notch=True, showmeans=True, showfliers=False)
plt.grid(True, alpha=0.5)
plt.savefig('other/alg_comparison/box-whisker-plot.png')
plt.close()

err1_conf = mean_confidence_interval(err1_abs)
err2_conf = mean_confidence_interval(err2_abs)
print('Errors 1 95% conf. interval: {}-{}'.format(err1_conf[1], err1_conf[2]))
print('Errors 2 95% conf. interval: {}-{}'.format(err2_conf[1], err2_conf[2]))
