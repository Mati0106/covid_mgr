import pandas as pd
import numpy as np
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# define path to data
p = PurePath(Path.cwd())
path = p.parents[1].joinpath('data').joinpath('nowy')

# load data
total_df = pd.read_csv(path.joinpath('total_df.csv'))
df = pd.read_csv(path.joinpath('total_df.csv'))


def mape(forecast, real):
    prediction = np.array(forecast)
    actual_value = np.array(real)
    results = np.mean(np.abs((actual_value - prediction) / actual_value)) * 100
    return results


# define country and signature
country = 'Poland'
country_p = 'PL'

# define variable
dep_variable = 'Recovered'
country_data = total_df[(total_df['Country/Region'] == str(country))].sort_values('Confirmed').reset_index(
    drop=True).sort_values(['Country/Region', 'Date']).drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
country_data['actual_cases'] = country_data['Confirmed'] - country_data['Deaths'] - country_data['Recovered']
actual_date = country_data[country_data['Date'] <= '2020-07-05'][
    ['Date', 'Confirmed', 'Recovered', 'Deaths', 'actual_cases', 'Country/Region']]

# plot acf for data
value = actual_date[dep_variable]
plot_acf(value)
plot_pacf(value)

# 2x run differencing
calc_diff = actual_date[dep_variable].copy().values
calc_diff[1:] -= calc_diff[:-1]
plt.plot(calc_diff)

# plot acf/pacf to define p,q
plot_acf(calc_diff)
plot_pacf(calc_diff)
plt.plot(calc_diff)

# choose parameters
# DEATHS
# param = (2,2,4)

# CONF
# params = (1,2,1)

# RECO
# params=(1,2,2)
params = (1, 2, 2)

# use dates till 05-05 to train
train = actual_date[:-62][dep_variable]
model_arima = ARIMA(train, params).fit()

# forecast 62 steps
forecast = model_arima.forecast(steps=62)[0]
results_may = train.copy().to_list()
results_may.extend(forecast)

# prepare visualizations
first_day = country_data.set_index('Date').index.values[0]
times = pd.date_range(first_day, periods=len(actual_date), freq='D')

fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(times,
        results_may,
        color='red',
        label='forecasted ' + dep_variable + ' cases')
ax.plot(times,
        actual_date[dep_variable],
        color='green',
        label='real ' + dep_variable + ' cases')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel=dep_variable,
       title="Forecast of " + dep_variable + " in Poland")
plt.axvline(times[-62], 0, 1, label='train/test split')
plt.legend()
plt.show()

# results for may
mape_may = mape(real=actual_date[-62:-31][dep_variable], forecast=forecast[:31])
mae_may = mean_absolute_error(actual_date[-62:-31][dep_variable], forecast[:31])
mse_may = mean_squared_error(actual_date[dep_variable][-62:-31], forecast[:31])

# results for june
mape_june = mape(real=actual_date[-31:][dep_variable], forecast=forecast[31:])
mae_june = mean_absolute_error(actual_date[dep_variable][-31:], forecast[31:])
mse_june = mean_squared_error(actual_date[dep_variable][-31:], forecast[31:])
