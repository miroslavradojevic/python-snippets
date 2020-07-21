#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.backends.backend_pdf import PdfPages

x = np.linspace(0, 50, 501)
y = np.sin(x)
df = pd.DataFrame(data=y, index=x, columns=['Sine'])
print(len(df))

test_percent = 0.1
test_point = np.round(len(df) * test_percent)
print(type(test_point), " ", test_point)

test_ind = int(len(df) - test_point)
print(type(test_ind))

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

print("train", type(train), " len=", len(train), " size=", train.size, " shape=", train.shape, " ndim=", train.ndim)
print("test", type(test), " len=", len(test), " size=", test.size, " shape=", test.shape, " ndim=", test.ndim)

scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

print("scaled_train", type(scaled_train), " len=", len(scaled_train), " size=", scaled_train.size, " shape=",
      scaled_train.shape, " ndim=", scaled_train.ndim)
print("scaled_test", type(scaled_test), " len=", len(scaled_test), " size=", scaled_test.size, " shape=",
      scaled_test.shape, " ndim=", scaled_test.ndim)

length = 3  # Length of the output sequences (in number of timesteps)
batch_size = 1  # Number of timeseries samples in each batch
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)

print("generator.length= ", len(generator))

X, y = generator[0]
print(type(X), X.shape, " -- ", type(y), y.shape)

length = 50  # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)
X, y = generator[0]
print(type(X), X.shape, " -- ", type(y), y.shape)

n_features = 1

model = Sequential()
model.add(SimpleRNN(50, input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator, epochs=5)

#####
first_eval_batch = scaled_train[-length:]
print("first_eval_batch", type(first_eval_batch), " len=", len(first_eval_batch), " size=", first_eval_batch.size,
      " shape=", first_eval_batch.shape, " ndim=", first_eval_batch.ndim)
first_eval_batch = first_eval_batch.reshape((1, length, n_features))
print("first_eval_batch", type(first_eval_batch), " len=", len(first_eval_batch), " size=", first_eval_batch.size,
      " shape=", first_eval_batch.shape, " ndim=", first_eval_batch.ndim)

test_predictions = []
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))
print("current_batch", type(current_batch), current_batch.shape)

c = current_batch[:, 1:, :]
print("current_batch[:,1:,:]", type(c), c.shape)

a = [[[99]]]
b = np.append(current_batch[:, 1:, :], a, axis=1)
print(type(a), len(a))
print(type(b), b.shape)

################################
test_predictions = []
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))
# print(current_batch.shape) # (1, length, 1)

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]  # numpy.ndarray (1, 1) into (1,)
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# print(type(test_predictions))
true_predictions = scaler.inverse_transform(test_predictions)  # list into ndarray (50, 1)
print(type(true_predictions), true_predictions.shape)

test['Predictions_RNN'] = true_predictions
# print(test.head)

early_stop = EarlyStopping(monitor='val_loss', patience=2)

length = 48

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)
# print("ok", scaled_test.shape)
validation_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=1)

model = Sequential()
model.add(LSTM(50, input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator, epochs=20, validation_data=validation_generator, callbacks=[early_stop])

test_predictions = []
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))
for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions_LSTM'] = true_predictions

# fig = plt.figure(figsize=(6, 4))
# plt.plot(test)
# fig = test.plot()
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
# plt.savefig('predictions.pdf', bbox_inches='tight')
# plt.close()

################
with PdfPages('predictions.pdf') as pdf:
    test.plot()
    pdf.savefig()
    plt.close()

full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)

length = 50  # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)

model = Sequential()
model.add(LSTM(50, input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator, epochs=6)

forecast = []
first_eval_batch = scaled_full_data[-length:]
print("scaled_full_data.shape=", scaled_full_data.shape)
print("first_eval_batch.shape=", first_eval_batch.shape)
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

forecast = scaler.inverse_transform(forecast)
print(df.tail())

forecast_index = np.arange(50.1,55.1,step=0.1)
fig = plt.figure(figsize=(6, 4))
plt.plot(df.index,df['Sine'])
plt.plot(forecast_index,forecast)
plt.show()
fig.savefig('forecast_0.pdf', bbox_inches='tight')
plt.close()

#concatenate
df1 = pd.DataFrame({'Sine':forecast.flatten()}, index=forecast_index)
df2 = pd.concat([df, df1], ignore_index=False)
fig = plt.figure(figsize=(6, 4))
plt.plot(df2)
# plt.plot(df2.mask(df2.apply(lambda x: x.index <= 50))[0], color='blue') # , figsize=(6, 4)
# plt.plot(df2.mask(df2.apply(lambda x: x.index > 50))[0], color='green')
plt.show()
fig.savefig('forecast_1.pdf')
plt.close()