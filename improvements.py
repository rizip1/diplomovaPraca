import numpy as np
import matplotlib.pyplot as plt
from utils import parse_hour, parse_month
from constants import IMPROVEMENT_PATH, COMPARED_IMPROVEMENTS_PATH


def _divide_improvements(improvements):
    morning = []
    afternoon = []
    for i, values in enumerate(improvements):
        mean = np.mean(values)
        if (i < 12):
            morning.append(mean)
        else:
            afternoon.append(mean)
    return (morning, afternoon)


def _get_hour_axis():
    return [(i + 1) for i in range(24)]


def _save_improvements_to_plots(improvements, seasonal_improvements):
    x = _get_hour_axis()
    morning, afternoon = _divide_improvements(improvements)

    plt.figure(figsize=(12, 6))
    plt.plot(x[0:12], morning, 'r')
    plt.plot(x[12:], afternoon, 'g')
    plt.title('Improvement')
    plt.ylabel('Errors')
    plt.xlabel('Hours')
    plt.xticks(x)
    plt.grid()
    plt.savefig('{}/total_improvement.png'.format(IMPROVEMENT_PATH))
    plt.close()

    plt.figure(figsize=(12, 6))
    colors = ['r', 'g', 'b', 'y']

    for index, period in enumerate(['spring', 'summer', 'autumn', 'winter']):
        morning, afternoon = _divide_improvements(
            seasonal_improvements[period])
        plt.plot(x[0:12], morning, colors[index], label=period)
        plt.plot(x[12:], afternoon, colors[index])

    plt.title('Seasonal improvement')
    plt.ylabel('Errors')
    plt.xlabel('Hours')
    plt.xticks(x)
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1.015), loc=2)
    plt.savefig('{}/seasonal_improvements.png'.format(IMPROVEMENT_PATH))
    plt.close()


def _get_improvements(data):
    improvements = [[] for i in range(24)]
    seasonal_improvements = {
        'spring': [[] for i in range(24)],
        'summer': [[] for i in range(24)],
        'autumn': [[] for i in range(24)],
        'winter': [[] for i in range(24)],
    }

    for i in data.index:
        val_date = data.loc[i, 'validity_date']
        hour = parse_hour(val_date)
        month = parse_month(val_date)

        y_test = data.loc[i, 'future_temp']
        shmu_value = data.loc[i, 'future_temp_shmu']
        y_predicted = data.loc[i, 'predicted']

        shmu_error = abs(y_test - shmu_value)
        predicted_error = abs(y_test - y_predicted)
        improvement = shmu_error - predicted_error

        p = hour - 1
        improvements[p].append(improvement)
        if (month in [1, 2, 3]):
            seasonal_improvements['winter'][p].append(improvement)
        elif (month in [4, 5, 6]):
            seasonal_improvements['spring'][p].append(improvement)
        elif (month in [7, 8, 9]):
            seasonal_improvements['summer'][p].append(improvement)
        elif (month in [10, 11, 12]):
            seasonal_improvements['autumn'][p].append(improvement)
    return (improvements, seasonal_improvements)


def _save_improvements_file(file_name, morning, afternoon, total_improvements,
                            total_worse, total_draws):
    with open('{}/{}.txt'.format(IMPROVEMENT_PATH, file_name), 'w') as f:
        f.write('Morning\n')
        for v in morning:
            f.write('{}\n'.format(v))

        f.write('\nAfternoon\n')
        for v in afternoon:
            f.write('{}\n'.format(v))

        f.write('\nTotal improvements: {}\n'.format(total_improvements))
        f.write('Total worse: {}\n'.format(total_worse))
        f.write('Total draws: {}\n'.format(total_draws))
        f.write('Total records: {}\n'.format(
                total_draws + total_improvements + total_worse))


def _classify_improvements(improvements):
    total_improvements = 0
    total_draws = 0
    total_worse = 0
    morning = []
    afternoon = []

    for index, values in enumerate(improvements):
        for v in values:
            if (v < 0):
                total_worse += 1
            elif (v > 0):
                total_improvements += 1
            else:
                total_draws += 1

        val = np.mean(values)
        if (index < 12):
            morning.append(val)
        else:
            afternoon.append(val)

    return (morning, afternoon, total_improvements, total_draws, total_worse)


def _save_improvements_to_file(improvements, seasonal_improvements):

    morning, afternoon, total_improvements, total_draws, \
        total_worse = _classify_improvements(improvements)

    _save_improvements_file('all_improvements', morning, afternoon,
                            total_improvements, total_worse, total_draws)

    for season, hour_values in seasonal_improvements.items():
        morning, afternoon, total_improvements, total_draws, \
            total_worse = _classify_improvements(hour_values)

        file_name = '{}_improvemets.txt'.format(season)
        _save_improvements_file(file_name, morning, afternoon,
                                total_improvements, total_worse, total_draws)


def compare_2_models_improvements(data1, data2):
    improvements1, seasonal_improvements1 = _get_improvements(data1)
    improvements2, seasonal_improvements2 = _get_improvements(data2)

    colors = ['b', 'g']
    x = _get_hour_axis()

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle('Seasonal improvement', fontsize=14)
    pos = [221, 222, 223, 224]
    for index, period in enumerate(['spring', 'summer', 'autumn', 'winter']):
        plt.subplot(pos[index])

        m1, a1 = _divide_improvements(seasonal_improvements1[period])
        m2, a2 = _divide_improvements(seasonal_improvements2[period])

        plt.plot(x[0:12], m1, colors[0], label='Model 1')
        plt.plot(x[0:12], m2, colors[1], label='Model 2')

        plt.plot(x[12:], a1, colors[0])
        plt.plot(x[12:], a2, colors[1])

        plt.title(period)
        plt.legend(loc=2)
        plt.ylabel('Improvement')
        plt.xlabel('Hours')
        plt.xticks(x)
        plt.grid()

    plt.tight_layout()
    plt.savefig('{}/comparison.png'.format(COMPARED_IMPROVEMENTS_PATH))
    plt.close()


def save_improvements(data):
    improvements, seasonal_improvements = _get_improvements(data)
    _save_improvements_to_plots(improvements, seasonal_improvements)
    _save_improvements_to_file(improvements, seasonal_improvements)
