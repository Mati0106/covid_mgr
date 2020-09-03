import pandas as pd
import numpy as np
from pathlib import Path, PurePath

from sklearn.metrics import mean_squared_error, mean_absolute_error

p = PurePath(Path.cwd())
path = p.parents[1].joinpath('data').joinpath('nowy')
total_df = pd.read_csv(path.joinpath('total_df.csv'))
df = pd.read_csv(path.joinpath('total_df.csv'))


def mape(forecast, real):
    prediction = np.array(forecast)
    actual_value = np.array(real)
    results = np.mean(np.abs((actual_value - prediction) / actual_value)) * 100
    return results


country = 'Poland'
country_p = 'PL'
dep_variable = 'Confirmed'
country_data = total_df[(total_df['Country/Region'] == str(country))].sort_values('Confirmed').reset_index(
    drop=True).sort_values(['Country/Region', 'Date']).drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
country_data['actual_cases'] = country_data['Confirmed'] - country_data['Deaths'] - country_data['Recovered']
actual_date = country_data[country_data['Date'] <= '2020-07-05'][
    ['Date', 'Confirmed', 'Recovered', 'Deaths', 'actual_cases', 'Country/Region']]

d_f = pd.read_csv(path.parent.joinpath(str(country_p) + '_deaths_final.csv'))
r_f = pd.read_csv(path.parent.joinpath(str(country_p) + '_reco_final.csv'))
c_f = pd.read_csv(path.parent.joinpath(str(country_p) + '_conf_final.csv'))

d_f['reco_actual_final'] = r_f['reco_actual_final']
d_f['conf_actual_final'] = c_f['conf_actual_final']
d_f['actual_cases'] = d_f['conf_actual_final'] - d_f['reco_actual_final'] - d_f['deaths_actual_final']

d_f['reco_predict_final'] = r_f['reco_predict_final']
d_f['conf_predict_final'] = c_f['conf_predict_final']
d_f['predict_cases'] = d_f['conf_predict_final'] - d_f['reco_predict_final'] - d_f['deaths_predict_final']

prediction = d_f[
    ['date', 'conf_predict_final', 'reco_predict_final', 'deaths_predict_final', 'predict_cases', 'country']].copy()
pred_test = prediction[-31:].copy()
pred_test.loc[pred_test['predict_cases'] <= 0, 'predict_cases'] = 0
actual_test = actual_date[-31:].copy()
actual_test.columns

# MSE
mse_conf = mean_squared_error(pred_test['conf_predict_final'], actual_test['Confirmed'])
mse_reco = mean_squared_error(pred_test['reco_predict_final'], actual_test['Recovered'])
mse_deaths = mean_squared_error(pred_test['deaths_predict_final'], actual_test['Deaths'])
mse_actual = mean_squared_error(pred_test['predict_cases'], actual_test['actual_cases'])

# MAE
mae_conf = mean_absolute_error(pred_test['conf_predict_final'], actual_test['Confirmed'])
mae_reco = mean_absolute_error(pred_test['reco_predict_final'], actual_test['Recovered'])
mae_deaths = mean_absolute_error(pred_test['deaths_predict_final'], actual_test['Deaths'])
mae_actual = mean_absolute_error(pred_test['predict_cases'], actual_test['actual_cases'])

# MAPE
mape_conf = mape(pred_test['conf_predict_final'], actual_test['Confirmed'])
mape_reco = mape(pred_test['reco_predict_final'], actual_test['Recovered'])
mape_deaths = mape(pred_test['deaths_predict_final'], actual_test['Deaths'])
mape_actual = mape(pred_test['predict_cases'], actual_test['actual_cases'])

print('results for ' + str(country))

d_f = pd.read_csv(path.parent.joinpath(str(country_p) + '_deaths_initial.csv'))
r_f = pd.read_csv(path.parent.joinpath(str(country_p) + '_reco_initial.csv'))
c_f = pd.read_csv(path.parent.joinpath(str(country_p) + '_conf_initial.csv'))

d_f['reco_actual_initial'] = r_f['reco_actual_initial']
d_f['conf_actual_initial'] = c_f['conf_actual_initial']
d_f['actual_cases'] = d_f['conf_actual_initial'] - d_f['reco_actual_initial'] - d_f['deaths_actual_initial']

d_f['reco_predict_initial'] = r_f['reco_predict_initial']
d_f['conf_predict_initial'] = c_f['conf_predict_initial']
d_f['predict_cases'] = d_f['conf_predict_initial'] - d_f['reco_predict_initial'] - d_f['deaths_predict_initial']

prediction = d_f[['date', 'conf_predict_initial', 'reco_predict_initial', 'deaths_predict_initial', 'predict_cases',
                  'country']].copy()
pred_test = prediction[-31:-24].copy()
pred_test.loc[pred_test['predict_cases'] <= 0, 'predict_cases'] = 0
actual_test = actual_date[-62:-55].copy()

# MSE
mse_conf = mean_squared_error(pred_test['conf_predict_initial'], actual_test['Confirmed'])
mse_reco = mean_squared_error(pred_test['reco_predict_initial'], actual_test['Recovered'])
mse_deaths = mean_squared_error(pred_test['deaths_predict_initial'], actual_test['Deaths'])
mse_actual = mean_squared_error(pred_test['predict_cases'], actual_test['actual_cases'])

# MAE
mae_conf = mean_absolute_error(pred_test['conf_predict_initial'], actual_test['Confirmed'])
mae_reco = mean_absolute_error(pred_test['reco_predict_initial'], actual_test['Recovered'])
mae_deaths = mean_absolute_error(pred_test['deaths_predict_initial'], actual_test['Deaths'])
mae_actual = mean_absolute_error(pred_test['predict_cases'], actual_test['actual_cases'])

# MAPE
mape_conf = mape(pred_test['conf_predict_initial'], actual_test['Confirmed'])
mape_reco = mape(pred_test['reco_predict_initial'], actual_test['Recovered'])
mape_deaths = mape(pred_test['deaths_predict_initial'], actual_test['Deaths'])
mape_actual = mape(pred_test['predict_cases'], actual_test['actual_cases'])

print('results for ' + str(country))
