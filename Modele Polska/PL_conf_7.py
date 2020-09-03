import pandas as pd
from pathlib import Path, PurePath
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from pandas import Series
import matplotlib.pyplot as plt
from keras.models import load_model
from numpy import array
from keras.callbacks import EarlyStopping
from matplotlib.dates import DateFormatter


# define function to split sequence
def split_sequence(input_sequence, n):
    X, y = list(), list()
    for i in range(len(input_sequence)):
        # last value
        end = i + n
        # check end
        if end > len(input_sequence) - 1:
            break
        # spliting sequence into x,y series
        seq_x, seq_y = input_sequence[i:end], input_sequence[end]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# set up path
p = PurePath(Path.cwd())
path = p.parents[1].joinpath('data')

# load data
total_df = pd.read_csv(path.joinpath('total_df.csv'))
#
# choose a number of time steps
n_steps = 7

# choose a number of features
n_features = 1

# define model
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))  #
model.add(LSTM(256, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# if model is already created uncomment that line and load it
# model = load_model('poland_conf_final.h5')

# choose dependent variable
dep_variable = 'Confirmed'
country_data = total_df[(total_df['Country/Region'] == str('Poland'))].sort_values('Confirmed').reset_index(
    drop=True).sort_values(['Country/Region', 'Date']).drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
data_to_lstm = pd.get_dummies(country_data[[dep_variable, 'Country/Region', 'Province/State']])
new_data = data_to_lstm[dep_variable].to_list()

# choose split
split = 31
train_data = new_data[:-split]
test_data = new_data[-split:]

# define scaler
scaler = MinMaxScaler()

# prepare data into RNN
X_train, y_train = split_sequence(train_data, n_steps)

# fit scaler on training data
scaler.fit(X_train)

# define scaler for dependent variable and fit it before model training
scaler_dep = MinMaxScaler()
series = Series(y_train)
values = series.values
values = values.reshape((len(values), 1))
scaler_dep.fit(values)

# chose early stopping
es = EarlyStopping(monitor='loss', mode='min', patience=10)

# begin loop for training model and to create forecast
for i in range(0, 31):  #
    X_train, y_train = split_sequence(train_data, n_steps)
    X_train = scaler.transform(X_train)
    series = Series(y_train)
    values_train = series.values
    values_train = values_train.reshape((len(values_train), 1))
    y_train = scaler_dep.transform(values_train)
    X_test, y_test = split_sequence(test_data, n_steps)
    X_test = scaler.transform(X_test)
    series = Series(y_test)
    values_test = series.values
    values_test = values_test.reshape((len(values_test), 1))
    y_test = scaler_dep.transform(values_test)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

    # if model is not fitted yet change on i==0
    if i == -1:
        model.fit(X_train, y_train, batch_size=64, epochs=400, verbose=2, shuffle=True
                  , callbacks=[es], validation_data=(X_test, y_test))
    else:
        print(i)
    x_input = array(X_train[-1])
    x_input = x_input.reshape((1, n_steps, n_features))

    # add results into train_data
    yhat = model.predict(x_input, verbose=0)
    yhat = scaler_dep.inverse_transform(yhat)
    train_data.extend(yhat[0])

# prepare visualizations
first_day = country_data.set_index('Date').index.values[0]
times = pd.date_range(first_day, periods=len(train_data), freq='D')
fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(times,
        train_data,
        color='red',
        label='forecasted confirmed cases')
ax.plot(times,
        country_data[dep_variable].values,
        color='green',
        label='real confirmed cases')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel=dep_variable,
       title="Forecast of " + dep_variable + " in Poland")
plt.axvline(times[-31], 0, 1, label='train/test split')
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.legend()
plt.show()

# save results
initial_results = pd.DataFrame(
    columns={'country', 'date', 'dependent_variable', 'conf_actual_initial', 'conf_predict_initial'},
    index=range(0, len(times)))
initial_results['country'] = 'Poland'
initial_results['date'] = times
initial_results['dependent_variable'] = dep_variable
initial_results['conf_actual_initial'] = country_data[dep_variable].values
initial_results['conf_predict_initial'] = train_data

initial_results.to_csv(path.joinpath('PL_conf_initial.csv'))

# prepare results on entire data set
train_data = new_data.copy()
test_data = new_data.copy()

X_train, y_train = split_sequence(train_data, n_steps)
scaler.fit(X_train)
scaler_dep = MinMaxScaler()
series = Series(y_train)
values = series.values
values = values.reshape((len(values), 1))
scaler_dep.fit(values)

for i in range(0, 31):  #

    X_train, y_train = split_sequence(train_data, n_steps)
    X_train = scaler.transform(X_train)
    series = Series(y_train)
    values_train = series.values
    values_train = values_train.reshape((len(values_train), 1))
    y_train = scaler_dep.transform(values_train)
    X_test, y_test = split_sequence(test_data, n_steps)
    X_test = scaler.transform(X_test)
    series = Series(y_test)
    values_test = series.values
    values_test = values_test.reshape((len(values_test), 1))
    y_test = scaler_dep.transform(values_test)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

    x_input = array(X_train[-1])
    x_input = x_input.reshape((1, n_steps, n_features))

    yhat = model.predict(x_input, verbose=0)
    yhat = scaler_dep.inverse_transform(yhat)
    train_data.extend(yhat[0])

# prepare visualizations
first_day = country_data.set_index('Date').index.values[0]
times = pd.date_range(first_day, periods=len(train_data), freq='D')

# Add x-axis and y-axis
fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(times,
        train_data,
        color='red',
        label='forecasted confirmed cases')
ax.plot(times[:-31],
        country_data[dep_variable].values,
        color='green',
        label='real confirmed cases')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel=dep_variable,
       title="Forecast of " + dep_variable + " in Poland")
plt.axvline(times[-31], 0, 1, label='train/test split')
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.legend()
plt.show()

# save results for entire data set
final_results = pd.DataFrame(
    columns={'country', 'date', 'dependent_variable', 'conf_actual_final', 'conf_predict_final'},
    index=range(0, len(times)))
final_results['country'] = 'Poland'
final_results['date'] = times
final_results['dependent_variable'] = dep_variable
final_results['conf_actual_final'] = country_data[dep_variable]
final_results['conf_predict_final'] = train_data

final_results.to_csv(path.joinpath('PL_conf_final.csv'))

# uncomment to save a model
# model.save('poland_conf_final.h5')
