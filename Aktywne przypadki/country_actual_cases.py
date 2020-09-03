import pandas as pd
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# define path to data
p = PurePath(Path.cwd())
path = p.parents[1].joinpath('data')

# load data
total_df = pd.read_csv(path.joinpath('total_df.csv'))

# set up country name, and signature
country = 'Poland'
country_p = 'PL'
country_data = total_df[(total_df['Country/Region'] == country)].sort_values('Confirmed').reset_index(
    drop=True).sort_values(['Country/Region', 'Date']).drop(['Unnamed: 0'], axis=1).reset_index(drop=True)

# load data with predictions for may
d_i = pd.read_csv(path.joinpath(str(country_p) + '_deaths_initial.csv'))
r_i = pd.read_csv(path.joinpath(str(country_p) + '_reco_initial.csv'))
c_i = pd.read_csv(path.joinpath(str(country_p) + '_conf_initial.csv'))

# calculation of actual cases from prediction and real actual cases
d_i['reco_actual_initial'] = r_i['reco_actual_initial']
d_i['conf_actual_initial'] = c_i['conf_actual_initial']
d_i['actual_cases'] = d_i['conf_actual_initial'] - d_i['reco_actual_initial'] - d_i['deaths_actual_initial']

d_i['reco_predict_initial'] = r_i['reco_predict_initial']
d_i['conf_predict_initial'] = c_i['conf_predict_initial']
d_i['predict_cases'] = d_i['conf_predict_initial'] - d_i['reco_predict_initial'] - d_i['deaths_predict_initial']

# prepare visualizations
first_day = country_data.set_index('Date').index.values[0]
times = pd.date_range(first_day, periods=len(d_i), freq='D')
fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(times,
        d_i['predict_cases'],
        color='red',
        label='forecasted active cases')
ax.plot(times,
        d_i['actual_cases'],
        color='green',
        label='real active cases')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel='actual_cases',
       title="Forecast of active cases in " + str(country))
plt.axvline(times[-31], 0, 1, label='train/test split')
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.legend()
plt.show()

# load forecast for june
d_f = pd.read_csv(path.joinpath(str(country_p) + '_deaths_final.csv'))
r_f = pd.read_csv(path.joinpath(str(country_p) + '_reco_final.csv'))
c_f = pd.read_csv(path.joinpath(str(country_p) + '_conf_final.csv'))

# calculation of forecasted actual cases and real actual cases
d_f['reco_actual_final'] = r_f['reco_actual_final']
d_f['conf_actual_final'] = c_f['conf_actual_final']
d_f['actual_cases'] = d_f['conf_actual_final'] - d_f['reco_actual_final'] - d_f['deaths_actual_final']

d_f['reco_predict_final'] = r_f['reco_predict_final']
d_f['conf_predict_final'] = c_f['conf_predict_final']
d_f['predict_cases'] = d_f['conf_predict_final'] - d_f['reco_predict_final'] - d_f['deaths_predict_final']

# prepare visualizations
first_day = country_data.set_index('Date').index.values[0]
times = pd.date_range(first_day, periods=len(d_f), freq='D')
fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(times,
        d_f['predict_cases'],
        color='red',
        label='forecasted active cases')
ax.plot(times,
        d_f['actual_cases'],
        color='green',
        label='real active cases')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel='actual_cases',
       title="Forecast of active cases in " + str(country))
plt.axvline(times[-31], 0, 1, label='train/test split')
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.legend()
plt.show()
