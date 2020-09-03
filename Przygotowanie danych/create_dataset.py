from pathlib import Path, PurePath
import pandas as pd
import numpy as np


def rolling_diff(data_frame, vars_to_calc, vars_to_groupby):
    core_df = data_frame.copy()
    if len(vars_to_groupby) == 1:
        vars_to_groupby = vars_to_groupby[0]
    for variable in vars_to_calc:
        core_df["new_cases_" + variable] = 0
        for country in core_df[vars_to_groupby[0]].drop_duplicates():
            for province in core_df[vars_to_groupby[1]][core_df[vars_to_groupby[0]] == str(country)].drop_duplicates():
                diff_val = core_df[variable][
                    (core_df[vars_to_groupby[0]] == country) & (core_df[vars_to_groupby[1]] == province)].values
                diff_val[1:] -= diff_val[:-1]
                core_df["new_cases_" + variable][
                    (core_df[vars_to_groupby[0]] == country) & (core_df[vars_to_groupby[1]] == province)] = diff_val
        core_df["new_cases_" + variable][core_df["new_cases_" + variable] <= 0] = 0
    return core_df


# define path to data
p = PurePath(Path.cwd())
path = p.parents[1].joinpath('data').joinpath('nowy')

# load data
wd_data = pd.read_csv(path.joinpath('worldometer_data.csv'))
covid_data = pd.read_csv(path.joinpath('covid_19_clean_complete.csv'))

# merge data
data = wd_data.merge(covid_data, how='inner', on='Country/Region')
data['ActiveCases'][data['Country/Region'] == str('Poland')].drop_duplicates()

# sort data
data_sorted = data[
    ['Date', 'Continent', 'Country/Region', 'Population', 'Province/State', 'Lat', 'Long', 'TotalCases',
     'Serious,Critical', 'Tot Cases/1M pop', 'Deaths/1M pop',
     'TotalTests', 'Tests/1M pop',
     'ActiveCases', 'Confirmed', 'Deaths', 'Recovered']].sort_values(
    ['Continent', 'Country/Region', 'Date']).reset_index(drop=True)

# create datatime
data_sorted['Date'] = pd.to_datetime(data_sorted['Date'])
data_sorted["Week"] = data_sorted['Date'].dt.week
data_sorted["Month"] = data_sorted['Date'].dt.month
data_sorted["Day"] = data_sorted['Date'].dt.day

# check nulls
data_sorted.isnull().sum()

# clear data from nulls
data_clear = data_sorted[(~pd.isnull(data_sorted['Continent'])) & (~pd.isnull(data_sorted['Population']))]

# check nulls again
data_clear.isnull().sum()

# replace on province if there is no province/state
data_clear['Province/State'] = data_clear['Province/State'].replace(np.nan, 'no_province/state')

# check nulls again
data_clear.isnull().sum()

# check first 50 rows of data
data_clear[data_clear['Country/Region'] == str('Poland')].head(50)

# calculation of diffs
data_new_cases = rolling_diff(data_clear, ['Confirmed', 'Deaths', 'Recovered'], ['Country/Region', 'Province/State'])
# save results
data_new_cases.to_csv(path.joinpath('data_new_cases.csv'))

# calculation of ActiveCases
data_new_cases.loc[pd.isnull(data_new_cases['ActiveCases']) == True, 'ActiveCases'] = data_new_cases['Confirmed'] - \
                                                                                      data_new_cases['Recovered'] - \
                                                                                      data_new_cases['Deaths']
# check nulls again
data_new_cases.isnull().sum()

# calculation of measures
data_new_cases['deaths_per_m'] = data_new_cases['Deaths'] / 1000000 * data_new_cases['Population'] / 1000000
data_new_cases['TotalTests'] = data_new_cases['TotalTests'].replace(np.nan, -1)
data_new_cases['Tests/1M pop'] = data_new_cases['Tests/1M pop'].replace(np.nan, -1)
data_new_cases['Deaths/1M pop'] = data_new_cases['Deaths/1M pop'].replace(np.nan, -1)
data_new_cases['Serious,Critical'] = data_new_cases['Serious,Critical'].replace(np.nan, -1)
total_sum_df = data_new_cases[['Continent', 'Country/Region', 'Recovered', 'Confirmed', 'Deaths']].groupby(
    ['Continent', 'Country/Region']).sum().reset_index().rename(columns={"Recovered": "Total_recovered",
                                                                         "Confirmed": "Total_confirmed",
                                                                         "Deaths": "Total_deaths"})
#
total_df = data_new_cases.merge(total_sum_df, how='inner', on=['Continent', 'Country/Region'])
total_df.isnull().sum()
#
total_df['Province/State'] = total_df['Province/State'].replace(np.nan, 'no_province/state')
total_df.isnull().sum()

# save final results
total_df.to_csv(path.joinpath('total_df.csv'))
