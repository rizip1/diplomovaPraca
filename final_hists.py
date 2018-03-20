import matplotlib.pyplot as plt
import numpy as np
import os
from constants import RESULTS_PATH

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# use `ensemble2` model without autoregressive term

mae_results = {
    11801: [1.1225, 1.7636],
    11803: [1.0093, 1.5941],
    11805: [0.9304, 1.4937],
    11812: [0.9229, 1.5937],
    11813: [0.8863, 1.0935],
    11816: [0.9048, 1.2525],
    11819: [0.9415, 1.3080],
    11826: [1.0371, 1.5626],
    11855: [0.9728, 1.4916],
    11856: [0.8944, 1.2120],
    11857: [1.0866, 2.1727],
    11858: [0.9248, 1.4007],
    11867: [1.0532, 2.5166],
    11878: [1.2424, 2.0274],
    11880: [1.0268, 1.8313],
    11900: [1.0283, 2.2467],
    11903: [1.0359, 2.0135],
    11916: [0.9700, 3.0407],
    11917: [1.1823, 2.2238],
    11918: [1.0448, 1.3369],
    11919: [1.2865, 2.6875],
    11927: [0.9700, 1.5608],
    11930: [1.0776, 4.0834],
    11933: [1.1031, 2.7813],
    11934: [1.1990, 2.1499],
    11938: [1.0149, 2.4320],
    11952: [1.0824, 2.1235],
    11958: [1.0842, 2.4577],
    11962: [1.0799, 2.0567],
    11963: [0.9854, 1.9434],
    11968: [0.9399, 1.5184],
    11976: [1.0222, 1.9237],
    11978: [1.0271, 1.5589],
    11993: [1.1592, 2.5500],
}

mse_results = {
    11801: [2.3212, 4.3580],
    11803: [1.8195, 3.8031],
    11805: [1.5932, 3.2649],
    11812: [1.7189, 3.9574],
    11813: [1.4788, 2.0425],
    11816: [1.5134, 2.4843],
    11819: [1.6000, 2.8711],
    11826: [1.9787, 3.6042],
    11855: [1.6978, 3.3844],
    11856: [1.4827, 2.4307],
    11857: [2.2157, 6.6629],
    11858: [1.5728, 3.1241],
    11867: [1.9574, 8.4315],
    11878: [2.8320, 6.1616],
    11880: [1.9205, 4.8315],
    11900: [1.9088, 7.1754],
    11903: [2.0108, 5.9388],
    11916: [1.8615, 12.4447],
    11917: [2.6131, 7.2302],
    11918: [1.9812, 3.0161],
    11919: [3.1144, 9.8747],
    11927: [1.8514, 3.8088],
    11930: [2.1879, 20.7126],
    11933: [2.2204, 10.4482],
    11934: [2.5729, 6.4523],
    11938: [1.8720, 7.8438],
    11952: [2.1235, 2.9957],
    11958: [3.0903, 10.1062],
    11962: [2.1723, 5.8907],
    11963: [1.7379, 5.3926],
    11968: [1.5984, 3.4764],
    11976: [1.9631, 5.3345],
    11978: [1.9690, 3.8804],
    11993: [2.4935, 9.0021],
}

mae_model = []
mae_aladin = []
mse_model = []
mse_aladin = []

for value in mae_results.values():
    mae_model.append(value[0])
    mae_aladin.append(value[1])

for value in mse_results.values():
    mse_model.append(value[0])
    mse_aladin.append(value[1])

plt.figure(figsize=(12, 6))
plt.boxplot([mae_model, mae_aladin, mse_model, mse_aladin], showfliers=False,
            labels=['MAE final', 'MAE Aladin', 'MSE final', 'MSE Aladin'])
plt.title('MAE and MSE comparison, ignoring outliers')
plt.savefig('{}/boxplot.png'.format(RESULTS_PATH))
plt.close()

plt.figure(figsize=(12, 6))
plt.hist(mae_model, rwidth=0.95)
plt.title('MAE for final model')
plt.xlabel('Absolute error')
plt.ylabel('Occurencies')
plt.savefig('{}/mae_errors.png'.format(RESULTS_PATH))
plt.close()

plt.figure(figsize=(12, 6))
plt.hist(mse_model, rwidth=0.95)
plt.title('MSE for final model')
plt.xlabel('Mean squared error')
plt.ylabel('Occurencies')
plt.savefig('{}/mse_errors.png'.format(RESULTS_PATH))
plt.close()

# calculate percentage improvement
mae_impr = []
mse_impr = []
for value in mae_results.values():
    mae_impr.append((value[1] - value[0]) / value[1])

for value in mse_results.values():
    mse_impr.append((value[1] - value[0]) / value[1])

plt.figure(figsize=(12, 6))
plt.hist(mae_impr, rwidth=0.95)
plt.title('Percentage improvement for MAE')
plt.xlabel('Improvement')
plt.ylabel('Occurencies')
plt.savefig('{}/mae_improvement.png'.format(RESULTS_PATH))
plt.close()

plt.figure(figsize=(12, 6))
plt.hist(mse_impr, rwidth=0.95)
plt.title('Percentage improvement for MSE')
plt.xlabel('Improvement')
plt.ylabel('Occurencies')
plt.savefig('{}/mse_improvement.png'.format(RESULTS_PATH))
plt.close()

print('MAE improvement: {0:.2f}% - {1:.2f}%'.format(
    100 * min(mae_impr), 100 * max(mae_impr)))
print('MSE improvement: {0:.2f}% - {1:.2f}%'.format(
    100 * min(mse_impr), 100 * max(mse_impr)))

print('MAE median: {0:.2f}%'.format(100 * np.median(mae_impr)))
print('MSE median: {0:.2f}%'.format(100 * np.median(mse_impr)))
