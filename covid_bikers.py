'''Bla bla'''

import pandas as pd
from datetime import datetime

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import brewer
from bokeh.transform import dodge
from bokeh.models import Range1d, SingleIntervalTicker, ColumnDataSource, NumeralTickFormatter

HYDE_PARK = [111, 153, 191, 213, 222, 248, 300, 303, 304, 406, 407]
KINGS_CROSS = [4, 14,34, 70, 431, 439, 593, 674]
WATERLOO = [154, 361, 374]
SOHO = [109, 129, 159, 192, 260, 383, 386]

def get_time_info(ftid, start_date, end_date):

    def _columnize(df):
        df.loc[:, 'weekday'] = df['t_lower'].apply(datetime.weekday)
        df.loc[:, 'day'] = df['t_lower'].apply(lambda x: x.day)
        df.loc[:, 'month'] = df['t_lower'].apply(lambda x: x.month)
        df.loc[:, 'tod'] = df['t_lower'].apply(lambda x: x.hour + x.minute / 60.0)
        return df

    def _slice(df, year):
        return df.loc[(df['t_lower'] >= datetime(year, *start_date)) & \
                      (df['t_lower'] < datetime(year, *end_date))]

    df_all_time = pd.read_csv(ftid, index_col=0)
    df_all_time['t_lower'] = pd.to_datetime(df_all_time['t_lower'])
    df_all_time['t_upper'] = pd.to_datetime(df_all_time['t_upper'])
    df_all_time.index.name = 'time_id'

    return _columnize(_slice(df_all_time, 2019)).drop(columns=['t_lower', 't_upper']), \
           _columnize(_slice(df_all_time, 2020)).drop(columns=['t_lower', 't_upper'])

def get_active_station_ids(df):

    df_station_count = df.groupby('station_id').count()
    max_c = df_station_count['departures'].max()
    df_active_stations = df_station_count.loc[df_station_count['departures'] == max_c]
    return set(df_active_stations.index.to_list())

def make_clean_df(fdata, ftid, start_date, end_date):

    time_2019, time_2020 = get_time_info(ftid, start_date, end_date)

    df_all_data = pd.read_csv(fdata)
    df_2019 = df_all_data.loc[df_all_data['time_id'].isin(time_2019.index)]
    df_2020 = df_all_data.loc[df_all_data['time_id'].isin(time_2020.index)]

    s_active = get_active_station_ids(df_2019).intersection(get_active_station_ids(df_2020))
    df_2019 = df_2019.loc[df_2019['station_id'].isin(s_active)]
    df_2019 = df_2019.set_index(['station_id', 'time_id'])
    df_2020 = df_2020.loc[df_2020['station_id'].isin(s_active)]
    df_2020 = df_2020.set_index(['station_id', 'time_id'])

    return df_2019.join(time_2019).reset_index(), df_2020.join(time_2020).reset_index()

def make_viz_line_all(df_2019, df_2020, station_ids=None):

    if not station_ids is None:
        df_2019 = df_2019.loc[df_2019['station_id'].isin(station_ids)]
        df_2019 = df_2019.groupby('time_id').agg({'arrivals': 'sum', 'departures': 'sum',
                                                  'weekday': 'mean', 'day': 'mean', 'month': 'mean', 'tod': 'mean'})
        df_2020 = df_2020.loc[df_2020['station_id'].isin(station_ids)]
        df_2020 = df_2020.groupby('time_id').agg({'arrivals': 'sum', 'departures': 'sum',
                                                  'weekday': 'mean', 'day': 'mean', 'month': 'mean', 'tod': 'mean'})

    # These filters are done so that first Monday of March aligns between 2019 and 2020
    df_2019 = df_2019.loc[(df_2019['month']==3) & (df_2019['day'] >= 4)]
    gg_19 = df_2019.groupby('time_id').agg({'departures':'sum','month':'mean','day':'mean'})
    df_2020 = df_2020.loc[(df_2020['month']==3) & (df_2020['day'] >= 2)]
    gg_20 = df_2020.groupby('time_id').agg({'departures':'sum','month':'mean','day':'mean'})
    gg_19 = gg_19.reset_index()
    gg_20 = gg_20.reset_index()


    colors = brewer['PRGn'][4]
    p = figure(plot_width=950, plot_height=300, toolbar_location='above', y_range=(0,2300), x_range=(-10,1344))
    p.line(gg_19.index, gg_19['departures'], color=colors[0], line_width=1)
    p.line(gg_20.index, gg_20['departures'], color=colors[-1], line_width=1)
    p.xaxis.visible = False
    p.yaxis.axis_label = '# Rentals in last 30 minutes'
    p.yaxis.axis_label_text_font = 'courier'
    p.yaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font = 'courier'

    show(p)
    raise RuntimeError


def make_viz_time_series(df_2019, df_2020, station_ids=None, weekday_group=None, weekday_of_interest='weekday'):

    if not station_ids is None:
        df_2019 = df_2019.loc[df_2019['station_id'].isin(station_ids)]
        df_2019 = df_2019.groupby('time_id').agg({'arrivals' : 'sum', 'departures' : 'sum',
                                                  'weekday' : 'mean', 'day' : 'mean', 'month' : 'mean', 'tod' : 'mean'})
        df_2020 = df_2020.loc[df_2020['station_id'].isin(station_ids)]
        df_2020 = df_2020.groupby('time_id').agg({'arrivals' : 'sum', 'departures' : 'sum',
                                                  'weekday' : 'mean', 'day' : 'mean', 'month' : 'mean', 'tod' : 'mean'})

    if not weekday_group is None:
        df_2019.loc[:, 'weekday_group'] = df_2019['weekday'].map(weekday_group)
        df_2020.loc[:, 'weekday_group'] = df_2020['weekday'].map(weekday_group)

    df_2019.loc[:, 'rental_events'] = df_2019['departures'] + df_2019['arrivals']
    df_2020.loc[:, 'rental_events'] = df_2020['departures'] + df_2020['arrivals']

    gg19 = df_2019.groupby(['weekday_group', 'tod']).mean()
    x_vals = gg19.loc[(weekday_of_interest, slice(None)), :].index.droplevel(0)
    y_vals_dep = gg19.loc[(weekday_of_interest, slice(None)), :]['departures']
    y_vals_arr = gg19.loc[(weekday_of_interest, slice(None)), :]['arrivals']
    y_vals_ren = gg19.loc[(weekday_of_interest, slice(None)), :]['rental_events']
    print (y_vals_ren.sum())

    print (df_2020)
    df_2020 = df_2020.loc[df_2020['weekday_group'] == weekday_of_interest]
    gg_20 = df_2020.groupby(['month','day'])

    colors = brewer['PRGn'][4]
    p = figure(plot_width=650, plot_height=500, toolbar_location='above', y_range=(0,240))
    p.line(x_vals, y_vals_ren, color=colors[0], line_width=5, line_alpha=1.0)
#    p.line(x_vals, y_vals_arr, color=colors[0], line_width=5, line_alpha=0.5, line_dash='dashed')
#    p.xaxis.axis_label = 'Time of Day'
#    p.xaxis.axis_label_standoff = 30
    p.xaxis.axis_label_text_font = 'courier'
    p.x_range = Range1d(4,23)
    p.xaxis.ticker = SingleIntervalTicker(interval=1)
    p.xaxis.major_label_text_font_size = "11pt"
    p.xaxis.major_label_text_font = 'courier'
    p.yaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font = 'courier'
    p.yaxis.axis_label = '# Rentals in last 30 minutes'
    p.yaxis.axis_label_text_font = 'courier'

    df_dummy = df_2019.loc[df_2019['weekday_group'] == weekday_of_interest]
    gg_20 = df_dummy.groupby(['month','day'])
    for day, day_data in gg_20:
        print (day, day_data['rental_events'].sum())
        p.line(day_data.tod, day_data.rental_events, color=colors[-1], line_width=1)
        if day in [(4,7),(4,9)]:
            p.line(day_data.tod, day_data.rental_events, color=colors[-1], line_width=3)

    #        p.line(day_data.tod, day_data.arrivals, color=colors[-1], line_width=1, line_dash='dashed')

    show(p)

def make_viz_totals(df_2019, df_2020, weekday_group):

    if not weekday_group is None:
        df_2019.loc[:, 'weekday_group'] = df_2019['weekday'].map(weekday_group)
        df_2020.loc[:, 'weekday_group'] = df_2020['weekday'].map(weekday_group)

    gg19_weekday = df_2019.loc[df_2019['weekday_group'].isin(['weekday', 'friday'])]
    gg19_weekday = gg19_weekday.groupby(['month','day']).sum()
    gg20_weekday = df_2020.loc[df_2020['weekday_group'].isin(['weekday', 'friday'])]
    gg20_weekday = gg20_weekday.groupby(['month','day']).sum()
    print (gg19_weekday[(2,4):(2,15)])

    d19 = [gg19_weekday[(2,1):(2,15)].mean()['departures'],
           gg19_weekday[(2,18):(2,28)].mean()['departures'],
           gg19_weekday[(3,1):(3,15)].mean()['departures'],
           gg19_weekday[(3,18):(3,29)].mean()['departures'],
           gg19_weekday[(4,1):(4,14)].mean()['departures']]
    d20 = [gg20_weekday[(2,1):(2,15)].mean()['departures'],
           gg20_weekday[(2,18):(2,29)].mean()['departures'],
           gg20_weekday[(3,3):(3,14)].mean()['departures'],
           gg20_weekday[(3,17):(3,31)].mean()['departures'],
           gg20_weekday[(4,1):(4,14)].mean()['departures']]
    print (d19, d20)

    data = {'time of year' : ['1st half Feb.', '2nd half Feb.', '1st half Mar.', '2nd half Mar.', '1st half Apr.'],
            '2019' : d19,
            '2020' : d20}
    source = ColumnDataSource(data)

    colors = brewer['PRGn'][4]
    print (colors)
    p = figure(plot_width=750, plot_height=500, toolbar_location='above', y_range=(0,31000),
               x_range = ['1st half Feb.', '2nd half Feb.', '1st half Mar.', '2nd half Mar.', '1st half Apr.'])
    p.vbar(x=dodge('time of year', -0.15, range=p.x_range), top='2019', width=0.3, source=source,
           color=colors[0], legend_label='2019 weekdays')
    p.vbar(x=dodge('time of year', 0.15, range=p.x_range), top='2020', width=0.3, source=source,
           color=colors[-1], legend_label='2020 weekdays')
    p.xaxis.major_label_text_font_size = "11pt"
    p.xaxis.major_label_text_font = 'courier'
    p.yaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font = 'courier'
    p.yaxis[0].formatter = NumeralTickFormatter(format="0a")
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.legend.visible = False

    show(p)

def main(f_data, f_timeid, start_date, end_date, station_subset, weekday_group, type_weekday):

    df_2019, df_2020 = make_clean_df(f_data, f_timeid, start_date, end_date)

#    make_viz_totals(df_2019, df_2020, weekday_group)
#    make_viz_line_all(df_2019, df_2020, station_subset)
    make_viz_time_series(df_2019, df_2020, station_subset, weekday_group, type_weekday)

if __name__ == '__main__':
    main(f_data='/Users/andersohrn/Development/london_bike_forecast/data_reformat_May21/1701_2004_30m/data.csv',
         f_timeid='/Users/andersohrn/Development/london_bike_forecast/data_reformat_May21/1701_2004_30m/tinterval.csv',
         start_date=(3, 1),
         end_date=(4, 15),
         station_subset=HYDE_PARK,
         weekday_group={0 : 'weekday', 1: 'weekday', 2: 'weekday', 3: 'weekday', 4: 'weekday', 5: 'weekend', 6: 'weekend'},
         type_weekday='weekend')