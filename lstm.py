import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# use 'validity_date' as index
dataset = pd.read_csv('data/data_11816.csv', delimiter=';', index_col=1)

'''
toDrop = ['reference_date',
          'current_temp', 'current_humidity',
          'current_pressure', 'current_rainfall_last_hour',
          'current_wind_speed', 'current_wind_direction',
          'p_time_pressure', 'p_time_humidity', 'p_time_rainfall_last_hour',
          'p_time_wind_speed', 'p_time_wind_direction']
'''

toDrop = ['reference_date',
          'current_temp', 'current_humidity',
          'current_pressure', 'current_rainfall_last_hour',
          'current_wind_speed', 'current_wind_direction',
          'p_time_rainfall_last_hour', 'p_time_wind_direction']

dataset.drop(toDrop, axis=1, inplace=True)
values = dataset.values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# split into train and test sets
train_length = 3000
train = scaled[:train_length, :]
test = scaled[train_length:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
# required by keras LSTM
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=100,
                    verbose=0, shuffle=False)

yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, :], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]

mse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = sqrt(mean_absolute_error(inv_y, inv_yhat))
print('Test MAE: %.3f' % mae)
print('Test MSE: %.3f' % mse)
