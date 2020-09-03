import pandas as pd
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from numpy import array


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


# define path to data
p = PurePath(Path.cwd())
path = p.parents[1].joinpath('data')

# load data
total_df = pd.read_csv(path.joinpath('total_df.csv'))

# define dep variable
dep_variable = 'Deaths'

# create visualizations
for continent in total_df['Continent'].drop_duplicates():
    Europe = total_df.copy()  # [total_df['Continent']==continent]
    Europe['Date'] = pd.to_datetime(Europe['Date'])
    Europe["Month_Name"] = Europe['Date'].dt.month_name()
    dep_variable = 'Recovered'  # 'Deaths' #'Confirmed' #'Recovered'
    Europe['m'] = Europe['Month_Name'].apply(
        lambda x: {'January': 0, 'February': 1, "March": 2, "April": 3, 'May': 4, 'June': 5, 'July': 6}[x])
    europe_plot = Europe.groupby(['Month_Name', 'm', 'Date']).agg(cumulative_recovered=('Recovered', 'sum'),
                                                                  cumulative_deaths=('Deaths', 'sum'),
                                                                  cumulative_confirmed=(
                                                                  'Confirmed', 'sum')).reset_index().sort_values(
        ["m", 'Date'])
    europe_plot_month = europe_plot.drop(['m', 'Date'], axis=1).set_index('Month_Name')
    fig, ax = plt.subplots()
    ax.plot(europe_plot_month['cumulative_confirmed'], label='Confirmed')
    ax.plot(europe_plot_month['cumulative_recovered'], label='Recovered')
    ax.plot(europe_plot_month['cumulative_deaths'], label='Deaths')
    plt.title('World month cumulative sum of cases')
    plt.xlabel('Month')
    plt.ylabel('Cumulative sum of cases')
    leg = ax.legend()
    plt.show()

# create daily visualizations
for continent in total_df['Continent'].drop_duplicates():
    Europe = total_df[total_df['Continent'] == continent].copy()  #
    Europe['Date'] = pd.to_datetime(Europe['Date'])
    Europe["Month_Name"] = Europe['Date'].dt.month_name()
    dep_variable = 'Recovered'  # 'Deaths' #'Confirmed' #'Recovered'
    Europe['m'] = Europe['Month_Name'].apply(
        lambda x: {'January': 0, 'February': 1, "March": 2, "April": 3, 'May': 4, 'June': 5, 'July': 6}[x])
    europe_plot = Europe.groupby(['Month_Name', 'm', 'Date']).agg(cumulative_recovered=('Recovered', 'sum'),
                                                                  cumulative_deaths=('Deaths', 'sum'),
                                                                  cumulative_confirmed=(
                                                                  'Confirmed', 'sum')).reset_index().sort_values(
        ["m", 'Date'])
    europe_plot_daily = europe_plot.drop(['Month_Name', 'm', 'Date'], axis=1).reset_index(drop=True)
    fig, ax = plt.subplots()
    ax.plot(europe_plot_daily['cumulative_confirmed'], label='Confirmed')
    ax.plot(europe_plot_daily['cumulative_recovered'], label='Recovered')
    ax.plot(europe_plot_daily['cumulative_deaths'], label='Deaths')
    plt.title(continent + ' daily cumulative sum of cases')
    plt.xlabel('Days')
    plt.ylabel('Cumulative sum of cases')
    leg = ax.legend()
    plt.show()

# create continent visualizations
for variable in ['Recovered', 'Deaths', 'Confirmed']:
    Europe = total_df.copy()
    Europe['Date'] = pd.to_datetime(Europe['Date'])
    Europe["Month_Name"] = Europe['Date'].dt.month_name()
    dep_variable = 'Recovered'  # 'Deaths' #'Confirmed' #'Recovered'
    Europe['m'] = Europe['Month_Name'].apply(
        lambda x: {'January': 0, 'February': 1, "March": 2, "April": 3, 'May': 4, 'June': 5, 'July': 6}[x])
    europe_plot = Europe.groupby(['Continent', 'Month_Name', 'm', 'Date']).agg(
        cumulative_recovered=(variable, 'sum')).reset_index().sort_values(["m", 'Date'])
    europe_plot_daily = europe_plot.drop(['Month_Name', 'm'], axis=1).reset_index(drop=True)
    all_continent_data = europe_plot_daily.sort_values(['Continent', 'Date'])
    fig, ax = plt.subplots()
    for continent in all_continent_data['Continent'].drop_duplicates():
        continent_data = all_continent_data[all_continent_data['Continent'] == str(continent)].reset_index(
            drop=True).drop('Date', axis=1)
        ax.plot(continent_data['cumulative_recovered'], label=continent)
        plt.title('World daily cumulative sum of ' + variable + ' cases')
        plt.xlabel('Days')
        plt.ylabel('Cumulative sum of cases')
        leg = ax.legend()
        plt.show()

# create increments visualizations
europe_plot = Europe.groupby(['Month_Name', 'm', 'Date']).agg(
    cumulative_deaths=(dep_variable, 'sum')).reset_index().sort_values(["m", 'Date'])
europe_plot_month = europe_plot.drop(['m', 'Date'], axis=1).set_index('Month_Name')
plt.plot(europe_plot_month)
plt.title('World month cumulative sum of ' + dep_variable + ' cases')
plt.xlabel('Month from Januray to early June ')
plt.ylabel('Cumulative sum of ' + dep_variable + ' cases')
plt.show()

europe_plot_daily = europe_plot.drop(['Month_Name', 'm', 'Date'], axis=1).reset_index(drop=True)
plt.plot(europe_plot_daily)
plt.title('World daily cumulative sum of ' + dep_variable + ' cases')
plt.xlabel('Days from 22-01-2020 to 04-06-2020 ')
plt.ylabel('Cumulative sum of ' + dep_variable + ' cases')
plt.show()

# weeks plot
europe_plot = Europe.groupby(['Week', 'Date']).agg(cumulative_deaths=(dep_variable, 'sum')).reset_index().sort_values(
    ['Date']).drop(['Date'], axis=1).set_index('Week')
plt.plot(europe_plot)
plt.title('Europe cumulative sum of ' + dep_variable)
