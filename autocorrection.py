import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
import csv
import math


def get_bias(real, predicted):
    r = real.astype(float)
    p = predicted.astype(float)
    return np.mean(r - p)


def show_figure(real_values, predicted_values, shmu_predicted):
    plt.figure(1)
    plt.plot(real_values, 'ob', label='Real values')
    plt.plot(predicted_values, 'or', label='Predicted values (Our model)')
    plt.plot(shmu_predicted, 'og', label='Predicted values (SHMU)')
    plt.legend(loc=1)
    plt.title('Temperature predictions')
    plt.ylabel('Temperature')
    plt.xlabel('Samples')
    plt.show()

data = []

with open('../data_all_hours_multiple_features_autocorrection.csv',
          newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    i = 0
    for row in reader:
        if (row):
            if (i != 0):
                data.append(row)
            i += 1

data = np.array(data)
y = data[:, -1:].astype(float)  # last column

# every column except last and first two columns which includes dates
X = data[:, 2:-1].astype(float)

m = X.shape[0]  # all data length
n = m // 2  # test set first start
train_set_len = m - n
mae_predict = 0
mse_predict = 0
mae_shmu = 0
mse_shmu = 0
predicted_all = []

lr = lm.LinearRegression(fit_intercept=True)
train_end = n
while (train_end < m):
    X_train = X[0:train_end, :]
    y_train = y[0:train_end]

    ref_date = data[train_end, 0]
    # check how many prediction we can make within same ref_date
    pred_length = 0
    while (data[train_end + pred_length, 0] == ref_date):
        pred_length += 1
        # out of bounds
        if (pred_length + train_end >= m):
            break

    X_test = X[train_end:train_end + pred_length, :]
    y_test = y[train_end:train_end + pred_length]
    # get weights
    weight = 0.98
    # change to train
    weights = list(reversed([math.sqrt(weight ** j)
                             for j in range(X_train.shape[0])]))
    weights = np.array(weights)

    lr.fit(X_train, y_train, sample_weight=weights)

    y_predicted = lr.predict(X_test)

    predicted_all.extend(list(y_predicted))

    mae_shmu += np.sum(abs(y_test.T - X_test[:, 4]))
    mse_shmu += np.sum((y_test.T - X_test[:, 4]) ** 2)

    mae_predict += np.sum(abs(y_test - y_predicted))
    mse_predict += np.sum((y_test - y_predicted) ** 2)

    # shift interval for learning
    train_end += pred_length


print('BIAS in whole data', get_bias(real=data[:, -1], predicted=data[:, -2]))
print('MAE SHMU', mae_shmu / train_set_len)
print('MAE PREDICT', mae_predict / train_set_len)
print('MSE SHMU', mse_shmu / train_set_len)
print('MSE PREDICT', mse_predict / train_set_len)

show_figure(y[-train_set_len:], np.array(predicted_all),
            X[-train_set_len:, -1])
