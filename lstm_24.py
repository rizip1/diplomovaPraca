import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.optimizers import SGD

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
start = 0
window_size = 24 * 30
train_end = window_size + 10000

data_len = train_end + 300

all_predicted = np.array([])
all_y_test = np.array([])

first_predictions = np.array([])
first_y_test = np.array([])

first_30_predictions = np.array([])
first_30_y_test = np.array([])

step = 24
predictions_made = 0
mae_predict = 0
mse_predict = 0

while (train_end < data_len):
    pred_length = step
    if (pred_length + train_end > data_len):
        pred_length = data_len - train_end

    print('Current position: ...', train_end)

    for i in range(pred_length):
        train_X = x.iloc[train_end - window_size +
                         i: train_end: step, :].values
        train_y = y.iloc[train_end - window_size +
                         i: train_end: step].values.reshape(-1, 1)

        test_X = x.iloc[train_end + i: train_end + pred_length: step, :].values
        test_y = y.iloc[train_end + i: train_end +
                        pred_length: step].values.reshape(-1, 1)

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

        print('training network {}'.format(i))
        model = Sequential()
        model.add(GRU(50, input_shape=(train_X.shape[1], train_X.shape[2]),
                      activation='tanh'))
        model.add(Dense(1))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mae', optimizer='sgd')
        history = model.fit(train_X, train_y, epochs=100, batch_size=4,
                            verbose=0, shuffle=False)
        yhat = model.predict(test_X)

        # free tensorFlow memory
        backend.clear_session()

        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

        # invert scaling for forecast
        y_predicted = get_predicted_values(test_X, yhat)

        # invert scaling for actual
        y_test = get_test_values(test_y, test_X)

        predictions_made += len(y_test)
        mae_predict += np.sum(abs(y_test - y_predicted))
        mse_predict += np.sum((y_test - y_predicted) ** 2)

        print('mse', mse_predict / predictions_made)
        print('mae', mae_predict / predictions_made)
        print('predictions count', predictions_made)

        all_y_test = np.append(all_y_test, y_test)
        all_predicted = np.append(all_predicted, y_predicted)

    # shift interval for learning
    train_end += pred_length
    start += pred_length

print('\nPredictions count', len(all_predicted))
print('MAE', mean_absolute_error(all_predicted, all_y_test))
print('MSE', mean_squared_error(all_predicted, all_y_test))
