import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import brewer
from bokeh.models import ColumnDataSource, NumeralTickFormatter, FixedTicker
import colorcet as cc
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram

from consts import EXCLUDE_STATIONS

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

def graph_info(df_gw, max_dist=1.0, weight_type='median_norm', lower_weight=None, common_weight=None, scale=1.0):

    def _map_color(vals, pallette):
        ret = []
        for s in vals:
            ind_palette = int((len(pallette) - 1) * s / max_dist)
            ret.append(pallette[ind_palette])
        return ret

    def _make_distance(x):
        return max_dist * np.exp(-1.0 * x * scale)

    df_gw = df_gw[['StartStation Id', 'EndStation Id', weight_type]]

    df_gw['StartStation Id'] = df_gw['StartStation Id'].astype(int)
    df_gw['EndStation Id'] = df_gw['EndStation Id'].astype(int)
    df_gw[weight_type] = df_gw[weight_type].astype(float)

    df_gw = df_gw.loc[~df_gw['StartStation Id'].isin(EXCLUDE_STATIONS)]
    df_gw = df_gw.loc[~df_gw['EndStation Id'].isin(EXCLUDE_STATIONS)]
    #TTT = [1,11,12,13,14,15]
    #df_gw = df_gw.loc[df_gw['StartStation Id'].isin(TTT)]
    #df_gw = df_gw.loc[df_gw['EndStation Id'].isin(TTT)]
    nstations = len(df_gw['StartStation Id'].unique())

    #df1 = df_gw.loc[df_gw['StartStation Id'] != df_gw['EndStation Id']].set_index(['StartStation Id', 'EndStation Id'])
    df1 = df_gw.set_index(['StartStation Id', 'EndStation Id'])
    df2 = df1.reorder_levels(['EndStation Id', 'StartStation Id'])
    df2 = df2.reindex(df1.index, fill_value=0.0)
    df3 = pd.concat([df1, df2], axis=1).mean(axis=1)
    df3 = df3.reset_index()

    if not lower_weight is None:
        df3 = df3.loc[df3[0] >= lower_weight]

    if not common_weight is None:
        df3[0] = common_weight

    df3['distance'] = df3[0].apply(_make_distance)
    df3['distance'] = np.where(df3['StartStation Id'] == df3['EndStation Id'], 0.0, df3['distance'])
    df_w = pd.pivot(df3, index='StartStation Id', columns='EndStation Id', values='distance').fillna(max_dist)

    clustering = AgglomerativeClustering(n_clusters=6, affinity='precomputed', linkage='average').fit(df_w.to_numpy())
    print (clustering.labels_)
#    clustering = SpectralClustering(n_clusters=6, affinity='precomputed').fit(df_w.to_numpy())
#    clustering = DBSCAN(metric='precomputed').fit(df_w.to_numpy())
    ss1 = pd.Series(clustering.labels_, df_w.index, name='cluster_id_s')
    ss2 = pd.Series(clustering.labels_, df_w.index, name='cluster_id_e')

    df_full = pd.DataFrame(df_w.stack())
    df_full = df_full.join(ss1, on='StartStation Id')
    df_full = df_full.join(ss2, on='EndStation Id')
    df_full = df_full.reset_index()

    clust_aa = df_full.groupby(['cluster_id_s','cluster_id_e']).mean().sort_values(by=0).index.to_list()
    gg = [x for x in clust_aa if x[0] != x[1]]
    print (gg)
    ppp = []
    for g in gg:
        g1 = g[0]
        g2 = g[1]
        if not g1 in ppp:
            ppp.append(g1)
        if not g2 in ppp:
            ppp.append(g2)
    ddd = dict([(val,key) for key, val in enumerate(ppp)])
    print (ddd)
    df_full['cluster_id_s'] = df_full['cluster_id_s'].replace(ddd)
    df_full['cluster_id_e'] = df_full['cluster_id_e'].replace(ddd)

    df_full = df_full.sort_values(['cluster_id_s', 'StartStation Id'])
    vals = []
    start_s = []
    end_s = []
    for nrow in range(nstations):
        hh = df_full.iloc[nrow * nstations:(nrow + 1) * nstations]
        hh = hh.sort_values(['cluster_id_e', 'EndStation Id'])
        vals.extend(hh[0].to_list())
        start_s.extend(hh['StartStation Id'].to_list())
        end_s.extend(hh['EndStation Id'].to_list())

    colors = _map_color(vals, list(cc.gray))
    #colors = _map_color(vals, list(reversed(cc.rainbow)))
    xx = list(range(nstations))
    data = dict(x=xx * nstations,
                y=sorted(xx * nstations, reverse=True),
                weight=vals,
                start_s=start_s,
                end_s=end_s,
                colors=colors)
    print (data)
    p = figure(x_range=(-1, nstations),
               y_range=(-1, nstations),
               tooltips=[('stations', '@start_s, @end_s'), ('value', '@weight')],
               width=800, height=800)
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.xaxis.major_label_text_font_size = '0pt'
    p.yaxis.major_label_text_font_size = '0pt'
    p.rect('x', 'y', 0.9, 0.9, source=data, color='colors', line_color=None)

    show(p)

def main(rawfile, interval_minutes, graphfile):

#    df = pd.read_csv(rawfile)
#    rental_dist_per_station(df, interval_minutes)

    df_g = pd.read_csv(graphfile)
    graph_info(df_g, lower_weight=1.0, weight_type='percent_flow', common_weight=1.0, scale=10.0)

if __name__ == '__main__':

    rawfile = '/Users/andersohrn/Development/london_bike_forecast/data_reformat_May21/1701_2004_15m/dataraw_30m.csv'
    graphfile = '/Users/andersohrn/Development/london_bike_forecast/data_reformat_May21/1701_2004_15m/graph_weight.csv'
    main(rawfile=rawfile, interval_minutes=30, graphfile=graphfile)