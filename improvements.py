import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import parse_hour, parse_month
from constants import IMPROVEMENT_PATH, COMPARED_IMPROVEMENTS_PATH


def divide_improvements(improvements):
    morning = []
    afternoon = []
    for i, values in enumerate(improvements):
        mean = np.mean(values)
        if (i < 12):
            morning.append(mean)
        else:
            afternoon.append(mean)
    return (morning, afternoon)


def get_hour_axis():
    return [(i + 1) for i in range(24)]


def save_improvements_to_plots(improvements, seasonal_improvements):
    x = get_hour_axis()
    morning, afternoon = divide_improvements(improvements)

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
        morning, afternoon = divide_improvements(seasonal_improvements[period])
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


def get_improvements(data):
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


def compare_2_models_improvements(data1, data2):
    improvements1, seasonal_improvements1 = get_improvements(data1)
    improvements2, seasonal_improvements2 = get_improvements(data2)

    colors = ['b', 'g']
    x = get_hour_axis()

    for index, period in enumerate(['spring', 'summer', 'autumn', 'winter']):
        plt.figure(figsize=(12, 6))
        m1, a1 = divide_improvements(seasonal_improvements1[period])
        m2, a2 = divide_improvements(seasonal_improvements2[period])

        plt.plot(x[0:12], m1, colors[0], label='Model 1')
        plt.plot(x[0:12], m2, colors[1], label='Model 2')

        plt.plot(x[12:], a1, colors[0])
        plt.plot(x[12:], a2, colors[1])

        plt.title('Seasonal improvement')
        plt.ylabel('Errors')
        plt.xlabel('Hours')
        plt.xticks(x)
        plt.grid()
        plt.legend(bbox_to_anchor=(1, 1.015), loc=2)
        plt.savefig('{}/{}.png'.format(COMPARED_IMPROVEMENTS_PATH, period))
        plt.close()


def save_improvements(data):
    improvements, seasonal_improvements = get_improvements(data)
    save_improvements_to_plots(improvements, seasonal_improvements)
