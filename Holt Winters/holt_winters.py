import pandas as pd
import numpy as np
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# define path to data
p = PurePath(Path.cwd())
path = p.parents[1].joinpath('data').joinpath('nowy')

# load core data
total_df = pd.read_csv(path.joinpath('total_df.csv'))
df = pd.read_csv(path.joinpath('total_df.csv'))


def mape(forecast, real):
    prediction = np.array(forecast)
    actual_value = np.array(real)
    results = np.mean(np.abs((actual_value - prediction) / actual_value)) * 100
    return results


# define country, signature and dep variable, change dep_variable to create results for others
country = 'Poland'
country_p = 'PL'
dep_variable = 'Deaths'
country_data = total_df[(total_df['Country/Region'] == str(country))].sort_values('Confirmed').reset_index(
    drop=True).sort_values(['Country/Region', 'Date']).drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
country_data['actual_cases'] = country_data['Confirmed'] - country_data['Deaths'] - country_data['Recovered']
actual_date = country_data[country_data['Date'] <= '2020-07-05'][
    ['Date', 'Confirmed', 'Recovered', 'Deaths', 'actual_cases', 'Country/Region']]

# holt-winters requiers to drop first 0 values
date_to_hw = actual_date[42:].reset_index(drop=True)

# set freq on index
actual_date.index.freq = 'D'

# train,test_may,test_june split
train, test, test_june = date_to_hw.iloc[:63, 3], date_to_hw.iloc[63:94, 3], date_to_hw.iloc[94:, 3]

# create model
model = ExponentialSmoothing(train, seasonal='add', trend='add', seasonal_periods=6).fit()

# create prediction for may
pred = model.predict(start=test.index[0], end=test.index[-1])

# create prediction for june
pred_june = model.predict(start=test.index[-1], end=124)

# summary of model (check AICC)
model.summary()

# prepare visualizations
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test_may')
plt.plot(test_june.index, test_june, label='Test_june')
plt.plot(pred.index, pred, label='Holt-Winters_may')
plt.plot(pred_june.index, pred_june, label='Holt-Winters_june')
plt.title(dep_variable + " in Poland")
plt.xlabel('Days')
plt.ylabel('Values')
plt.legend(loc='best')

# results calculation
mape_may = mape(real=test, forecast=pred)
mae_may = mean_absolute_error(test, pred)
mse_may = mean_squared_error(test, pred)

mape_june = mape(real=test_june, forecast=pred_june[1:-1])
mae_june = mean_absolute_error(test_june, pred_june[1:-1])
mse_june = mean_squared_error(test_june, pred_june[1:-1])
