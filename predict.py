import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
import csv


def get_bias(real, predicted):
    return np.mean(real - predicted)


def show_figure(real_values, predicted_values, shmu_predicted):
    plt.figure(1)
    plt.plot(real_values, 'ok', label='Real values')
    plt.plot(predicted_values, 'or', label='Predicted values (Our model)')
    plt.plot(shmu_predicted, 'og', label='Predicted values (SHMU)')
    plt.legend(loc=1)
    plt.title('Temperature predictions')
    plt.ylabel('Temperature')
    plt.xlabel('Samples')
    plt.show()

data = []

with open('../data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    i = 0
    for row in reader:
        if (row):
            if (i != 0):
                data.append(row)
            i += 1

# retype everything to float
data = np.array(data).astype(float)

y = data[:, -1:]  # last column
X = data[:, 0:data.shape[1] - 1]  # every column except last

m = X.shape[0]  # all data length
n = m // 2  # test set first start
train_set_len = m - n
mae_predict = 0
mse_predict = 0
mae_shmu = 0
mse_shmu = 0
predicted_all = []

# Sklearn by default adds intercept term to data.
# However it do not include it into lr.coef_ params.
# If we want to get all theta values we need to
# include column of ones and set fit_intercept=False.
lr = lm.LinearRegression(fit_intercept=True)
for train_end in range(n, m):
    X_train = X[0:train_end, :]
    X_test = X[train_end, :].reshape(1, -1)  # has to be 2D for numpy
    y_train = y[0:train_end]
    y_test = y[train_end]

    lr.fit(X_train, y_train)

    y_predicted = lr.predict(X_test)
    predicted_all.append(float(y_predicted))

    mae_shmu += float(abs(y_test - X_test[0, 1]))
    mse_shmu += float((y_test - X_test[0, 1]) ** 2)

    mae_predict += float(abs(y_test - y_predicted))
    mse_predict += float((y_test - y_predicted) ** 2)

print('BIAS in whole data', get_bias(data[:, 2], data[:, 1]))
print('MAE SHMU', mae_shmu / train_set_len)
print('MAE PREDICT', mae_predict / train_set_len)
print('MSE SHMU', mse_shmu / train_set_len)
print('MSE PREDICT', mse_predict / train_set_len)

show_figure(X[-train_set_len:, -1:], np.array(predicted_all),
            X[-train_set_len:, 0])