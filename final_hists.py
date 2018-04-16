import matplotlib.pyplot as plt
import numpy as np
import os
from utils import color_print
from constants import RESULTS_PATH

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# [0] - ref
# [1] - `ensemble2` model without autoregressive term
# [2] - aladin
# [3] - hum

mae_results = {
    11801: [1.1524, 1.1225, 1.7636, 1.1516],
    11803: [1.0358, 1.0093, 1.5941, 1.0289],
    11805: [0.9549, 0.9304, 1.4937, 0.9439],
    11812: [0.9782, 0.9229, 1.5937, 0.9587],
    11813: [0.9068, 0.8863, 1.0935, 0.8964],
    11816: [0.9438, 0.9048, 1.2525, 0.9323],
    11819: [0.9779, 0.9415, 1.3080, 0.9543],
    11826: [1.0719, 1.0371, 1.5626, 1.0574],
    11855: [1.0090, 0.9728, 1.4916, 0.9939],
    11856: [0.9250, 0.8944, 1.2120, 0.9036],
    11857: [1.1341, 1.0866, 2.1727, 1.1207],
    11858: [0.9685, 0.9248, 1.4007, 0.9514],
    11867: [1.0795, 1.0532, 2.5166, 1.0737],
    11878: [1.3065, 1.2424, 2.0274, 1.2862],
    11880: [1.0755, 1.0268, 1.8313, 1.0518],
    11900: [1.0715, 1.0283, 2.2467, 1.0568],
    11903: [1.0830, 1.0359, 2.0135, 1.0613],
    11916: [0.9949, 0.9700, 3.0407, 0.9856],
    11917: [1.2442, 1.1823, 2.2238, 1.2045],
    11918: [1.0785, 1.0448, 1.3369, 1.0544],
    11919: [1.3324, 1.2865, 2.6875, 1.3200],
    11927: [1.0077, 0.9700, 1.5608, 0.9976],
    11930: [1.1301, 1.0776, 4.0834, 1.0927],
    11933: [1.1384, 1.1031, 2.7813, 1.1177],
    11934: [1.2254, 1.1990, 2.1499, 1.2134],
    11938: [1.0431, 1.0149, 2.4320, 1.0273],
    11952: [1.1188, 1.0824, 2.1235, 1.1010],
    11958: [1.1146, 1.0842, 2.4577, 1.1094],
    11962: [1.1388, 1.0806, 2.0567, 1.1134],
    11963: [1.0161, 0.9854, 1.9434, 1.0013],
    11968: [0.9695, 0.9399, 1.5184, 0.9530],
    11976: [1.0618, 1.0222, 1.9237, 1.0394],
    11978: [1.0674, 1.0271, 1.5589, 1.0396],
    11993: [1.1986, 1.1592, 2.5500, 1.1829],
}

mse_results = {
    11801: [2.4026, 2.3212, 4.3580, 2.3985],
    11803: [1.8732, 1.8195, 3.8031, 1.8660],
    11805: [1.6244, 1.5932, 3.2649, 1.5987],
    11812: [1.8674, 1.7189, 3.9574, 1.8078],
    11813: [1.5039, 1.4788, 2.0425, 1.4916],
    11816: [1.5930, 1.5134, 2.4843, 1.5715],
    11819: [1.6758, 1.6000, 2.8711, 1.6142],
    11826: [2.0497, 1.9787, 3.6042, 2.0172],
    11855: [1.7921, 1.6978, 3.3844, 1.7350],
    11856: [1.5464, 1.4827, 2.4307, 1.4986],
    11857: [2.3403, 2.2157, 6.6629, 2.2918],
    11858: [1.6688, 1.5728, 3.1241, 1.6172],
    11867: [2.0275, 1.9574, 8.4315, 2.0038],
    11878: [3.0926, 2.8320, 6.1616, 2.9734],
    11880: [2.0703, 1.9205, 4.8315, 1.9804],
    11900: [2.0162, 1.9088, 7.1754, 1.9682],
    11903: [2.1261, 2.0108, 5.9388, 2.0555],
    11916: [2.0140, 1.8615, 12.4447, 1.9214],
    11917: [2.8933, 2.6131, 7.2302, 2.7180],
    11918: [2.0748, 1.9812, 3.0161, 1.9981],
    11919: [3.2751, 3.1144, 9.8747, 3.2289],
    11927: [1.9495, 1.8514, 3.8088, 1.9128],
    11930: [2.4156, 2.1879, 20.7126, 2.2470],
    11933: [2.3729, 2.2204, 10.4482, 2.2735],
    11934: [2.7276, 2.5729, 6.4523, 2.6330],
    11938: [1.9977, 1.8720, 7.8438, 1.9248],
    11952: [2.2452, 2.1235, 2.9957, 2.1607],
    11958: [3.2513, 3.0903, 10.1062, 3.2168],
    11962: [2.3491, 2.1736, 5.8907, 2.2509],
    11963: [1.8450, 1.7379, 5.3926, 1.7945],
    11968: [1.6718, 1.5984, 3.4764, 1.6239],
    11976: [2.0456, 1.9631, 5.3345, 1.9729],
    11978: [2.0975, 1.9690, 3.8804, 2.0012],
    11993: [2.5933, 2.4935, 9.0021, 2.5428],
}


def compare(mae_results, mse_results, exp_greater, exp_lower, title,
            mae_path, mse_path):
    '''
    calculate percentage improvement
    '''
    color_print(title)
    mae_impr = []
    mse_impr = []
    for value in mae_results.values():
        mae_impr.append(((value[exp_greater] - value[exp_lower]) /
                         value[exp_greater]) * 100)

    for value in mse_results.values():
        mse_impr.append(((value[exp_greater] - value[exp_lower]) /
                         value[exp_greater]) * 100)

    print('MAE improvement: {0:.2f}% - {1:.2f}%'.format(
        min(mae_impr), max(mae_impr)))
    print('MSE improvement: {0:.2f}% - {1:.2f}%'.format(
        min(mse_impr), max(mse_impr)))

    print('MAE median: {0:.2f}%'.format(np.median(mae_impr)))
    print('MSE median: {0:.2f}%'.format(np.median(mse_impr)))

    plt.figure(figsize=(12, 6))
    plt.hist(mae_impr, rwidth=0.95)
    plt.title('Percentage improvement for MAE')
    plt.xlabel('Percentage improvement')
    plt.ylabel('Occurrencies')
    plt.savefig('{}/{}.png'.format(RESULTS_PATH, mae_path))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(mse_impr, rwidth=0.95)
    plt.title('Percentage improvement for MSE')
    plt.xlabel('Percentage improvement')
    plt.ylabel('Occurrencies')
    plt.savefig('{}/{}.png'.format(RESULTS_PATH, mse_path))
    plt.close()


mae_model = []
mae_aladin = []
mse_model = []
mse_aladin = []
mae_ref = []
mse_ref = []
mae_hum = []
mse_hum = []

for value in mae_results.values():
    mae_ref.append(value[0])
    mae_model.append(value[1])
    mae_aladin.append(value[2])
    mae_hum.append(value[3])

for value in mse_results.values():
    mse_ref.append(value[0])
    mse_model.append(value[1])
    mse_aladin.append(value[2])
    mse_hum.append(value[3])

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

compare(mae_results, mse_results, 2, 1, 'Aladin to Final comparison',
        'mae_improvement', 'mse_improvement')

#####################################################################

compare(mae_results, mse_results, 0, 1, '\nRef to Final comparision',
        'mae_improvement_ref_to_final', 'mse_improvement_ref_to_final')

plt.figure(figsize=(12, 6))
plt.boxplot([mae_model, mae_ref, mse_model, mse_ref], showfliers=False,
            labels=['MAE final', 'MAE ref', 'MSE final', 'MSE ref'])
plt.title('MAE and MSE comparison, ignoring outliers')
plt.savefig('{}/boxplot_ref_to_final.png'.format(RESULTS_PATH))
plt.close()

plt.figure(figsize=(12, 6))
plt.boxplot([mae_model, mae_ref], showfliers=False,
            labels=['MAE final', 'MAE ref'])
plt.title('MAE comparison, ignoring outliers')
plt.savefig('{}/boxplot_mae_ref_to_final.png'.format(RESULTS_PATH))
plt.close()

plt.figure(figsize=(12, 6))
plt.boxplot([mse_model, mse_ref], showfliers=False,
            labels=['MSE final', 'MSE ref'])
plt.title('MSE comparison, ignoring outliers')
plt.savefig('{}/boxplot_mse_ref_to_final.png'.format(RESULTS_PATH))
plt.close()

#####################################################################

compare(mae_results, mse_results, 3, 1, '\nRef to Hum',
        'mae_improvement_ref_to_hum', 'mse_improvement_ref_to_hum')
