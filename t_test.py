import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
from statsmodels.stats.stattools import durbin_watson

from scipy.stats import ttest_rel, kstest, norm, wilcoxon

# kstest (Kolmogorov-Smirnov test)


def mean_confidence_interval(data, confidence=0.95):
    # TODO add stack-overflow source
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

err1 = np.squeeze(pd.read_csv('other/compare_errors/errors_60_24.csv').values)
err2 = np.squeeze(pd.read_csv(
    'other/compare_errors/errors_90_24.csv').values)

samples_count = min(err1.shape[0], err2.shape[0])

err1 = err1[-samples_count:]
err2 = err2[-samples_count:]

err1_abs = np.absolute(err1)
err2_abs = np.absolute(err2)

print('Durbin-watson1', durbin_watson(err1 - err2))
print('Durbin-watson2', durbin_watson(err1_abs - err2_abs))

dist1 = err1 - err2
dist2 = err1_abs - err2_abs
mu1, std1 = norm.fit(dist1)
mu2, std2 = norm.fit(dist2)
print('Kol-Sch1', kstest((dist1 - mu1) / std1, 'norm').pvalue)
print('Kol-Sch2', kstest((dist2 - mu2) / std2, 'norm').pvalue)

plt.figure(figsize=(12, 6))
plt.hist(dist1, bins=100, normed=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu1, std1)
plt.plot(x, p, 'k', linewidth=2)
plt.savefig('other/alg_comparison/hist1.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.hist(dist2, bins=100, normed=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu2, std2)
plt.plot(x, p, 'k', linewidth=2)
plt.savefig('other/alg_comparison/hist2.png')
plt.close()

print('Compared on {} samples'.format(samples_count))

# t-test for dependent x and y
print('Paired welch t-test', ttest_rel(err1_abs, err2_abs).pvalue)

# print('Paired Wilcoxon 1', wilcoxon(err1, err2).pvalue)
print('Paired Wilcoxon', wilcoxon(err1_abs, err2_abs).pvalue)

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
