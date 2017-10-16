import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from keras.constraints import maxnorm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU
from random import randint

from feature_utils import add_min_max
from feature_utils import add_moments
from feature_utils import shmu_prediction_time_error


def get_predicted_values(test_x, yhat):
    inv_yhat = np.concatenate((test_x, yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    return inv_yhat[:, -1]


def get_test_values(test_y, test_x):
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_x, test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    return inv_y[:, -1]

dataset = pd.read_csv('data/data_11816.csv', delimiter=';')

dataset = shmu_prediction_time_error(dataset, 1, 1, 0)

dataset = add_moments(dataset, 'mean')

dataset = add_min_max(dataset, 'min')

to_drop = ['reference_date', 'validity_date',
           'current_temp', 'current_humidity',
           'current_pressure', 'current_rainfall_last_hour',
           'current_wind_speed', 'current_wind_direction',
           'p_time_wind_speed', 'p_time_pressure', 'p_time_humidity',
           'p_time_rainfall_last_hour', 'p_time_wind_direction']

dropped_data = dataset.drop(to_drop, axis=1)

x = dropped_data.drop('future_temp', axis=1)
y = dropped_data.future_temp

data_len = dataset.shape[0]
window_len = 24 * 120

all_predicted = np.array([])
all_y_test = np.array([])

interval = 24
start = window_len
predictions_made = 0
mae_predict = 0
mse_predict = 0

total_predictions = 1000

linear_mae = 0
linear_mse = 0
neural_mae = 0
neural_mse = 0

reg_better = 0
nn_better = 0

for j in range(total_predictions):
    i = randint(start, data_len - 1)

    train_X = x.iloc[i - start:i:interval, :].values
    train_y = y.iloc[i - start:i:interval].values.reshape(-1, 1)

    test_X = x.iloc[i, :].values.reshape(1, -1)
    test_y = y.iloc[i].reshape(-1, 1)

    train_X_orig = train_X.copy()
    train_y_orig = train_y.copy()
    test_X_orig = test_X.copy()
    test_y_orig = test_y.copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(
        np.concatenate((train_X, train_y), axis=1))

    train_X = scaled[:, :-1]
    train_y = scaled[:, -1]

    tmp = scaler.transform(np.concatenate((test_X, test_y), axis=1))
    test_X = tmp[:, :-1]
    test_y = tmp[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(GRU(300, input_shape=(train_X.shape[1], train_X.shape[2]),
                  activation='tanh', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='RMSprop')
    history = model.fit(train_X, train_y, epochs=100, batch_size=10,
                        verbose=0, shuffle=False)
    yhat = model.predict(test_X)

    # free tensorFlow memory
    backend.clear_session()

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    linear_model = lm.LinearRegression(fit_intercept=True)
    linear_model.fit(train_X_orig, train_y_orig)
    linear_prediction = linear_model.predict(test_X_orig)

    # invert scaling for forecast
    y_predicted = get_predicted_values(test_X, yhat)

    # invert scaling for actual
    y_test = get_test_values(test_y, test_X)

    predictions_made += len(y_test)

    neural_mae += np.sum(abs(y_test - y_predicted))
    neural_mse += np.sum((y_test - y_predicted) ** 2)

    linear_mae += np.sum(abs(y_test - linear_prediction))
    linear_mse += np.sum((y_test - linear_prediction) ** 2)

    if (abs(y_test - linear_prediction) < abs(y_test - y_predicted)):
        reg_better += 1

    if (abs(y_test - linear_prediction) > abs(y_test - y_predicted)):
        nn_better += 1

    print('\nmae-lin-neu', linear_mae / predictions_made,
          neural_mae / predictions_made)
    print('mse-lin-neu', linear_mse / predictions_made,
          neural_mse / predictions_made)

    y_predicted = (y_predicted + linear_prediction) / 2

    mae_predict += np.sum(abs(y_test - y_predicted))
    mse_predict += np.sum((y_test - y_predicted) ** 2)

    print('mae', mae_predict / predictions_made)
    print('mse', mse_predict / predictions_made)
    print('predictions count', predictions_made)

    print('\nreg_better', reg_better)
    print('nn_better', nn_better)

    all_y_test = np.append(all_y_test, y_test)
    all_predicted = np.append(all_predicted, y_predicted)

print('\nPredictions count', len(all_predicted))
print('MAE', mean_absolute_error(all_predicted, all_y_test))
print('MSE', mean_squared_error(all_predicted, all_y_test))
