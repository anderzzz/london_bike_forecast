import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import brewer
from bokeh.models import ColumnDataSource, NumeralTickFormatter, FixedTicker

def rental_dist_per_station(df, interval_minutes, temporal='week'):

    if temporal == 'week':
        t_size = 60 * 24 * 7 // interval_minutes
    elif temporal == 'day':
        t_size = 60 * 24 // interval_minutes

    t_min = df['time_id'].min()
    df.loc[:, 'week_index'] = df.loc[:, 'time_id'].apply(lambda x: (x - t_min) // t_size)
    station_groups = df.groupby(['station_id', 'week_index']).sum()
    station_groups['tot'] = station_groups['arrivals'] + station_groups['departures']
    weekly = station_groups.groupby('station_id').mean()
    weekly['count_bin'], bin_vals = pd.cut(weekly['tot'], bins=range(0,3500,50), retbins=True)
    weekly = weekly.reset_index()
    weekly['mid_point'] = weekly['count_bin'].apply(lambda x: int(x.mid))
    histogram = weekly.groupby('mid_point').count().reset_index()

    source = ColumnDataSource(histogram)
    colors = brewer['PRGn'][4]
    p = figure(plot_width=800, plot_height=400, toolbar_location='above', y_range=(0,80), x_range=(0, 3400))
    p.vbar(x='mid_point', top='tot', source=source, fill_color=colors[0], line_color=colors[-1], width=50)
    p.xaxis.axis_label = 'average weekly departures+arrivals at station'
    p.yaxis.axis_label = '# stations'
    #show(p)

    print (station_groups)
    dd = station_groups.reset_index().groupby('week_index').sum()
    dd = dd.loc[:52 * 3,:]
    x = dd.index.to_list()
    y = dd['departures'].to_list()
    source = ColumnDataSource({'x':x,'y':y})
    p = figure(plot_width=800, plot_height=400, toolbar_location='above', x_range=(0,52*3))
    p.line(x='x', y='y', source=source, color=colors[0], line_width=3)
    p.xaxis.axis_label = 'weeks since 1st January 2017'
    p.xaxis.ticker = FixedTicker(ticks=list(range(0,160,20)))
    p.yaxis.axis_label = '# rental events in week'
    p.yaxis.formatter = NumeralTickFormatter(format='0a')
    #show(p)

    df_x = df.loc[df['week_index'] == 20]
    tt = df_x.groupby('time_id').sum()
    x = tt.index
    y = tt['departures'].to_list()
    source = ColumnDataSource({'x':x,'y':y})
    p = figure(plot_width=800, plot_height=400, toolbar_location='above', y_range=(0,2600))
    p.xgrid.grid_line_alpha = 0.0
    p.line(x='x',y='y', source=source, color=colors[0], line_width=3)
    p.yaxis.axis_label = '# rentals in half hour'
    p.xaxis.ticker = FixedTicker(ticks=list(range(8040,8350,48)))
    p.xaxis.major_label_overrides = {8040:'Wednesday', 8088:'Thursday', 8136:'Friday',8184:'Saturday',8232:'Sunday',8280:'Monday',8328:'Tuesday'}
    show(p)

def main(rawfile, interval_minutes):

    df = pd.read_csv(rawfile)

    rental_dist_per_station(df, interval_minutes)

if __name__ == '__main__':

    rawfile = '/Users/andersohrn/Development/london_bike_forecast/data_reformat_May21/1701_2004_30m/dataraw_30m.csv'
    main(rawfile=rawfile, interval_minutes=30)