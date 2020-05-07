import pandas as pd
from datetime import datetime, time, timedelta

def make_time_interval(start_date, end_date, interval_size):

    x0 = datetime.strptime(start_date, '%Y/%m/%d')
    x1 = datetime.strptime(end_date, '%Y/%m/%d')
    time_jump = timedelta(minutes=interval_size)

    t_interval_id = 0
    t_outer_bound = x0
    vals = []
    while t_outer_bound < x1:
        t_inner_bound = t_outer_bound
        t_outer_bound += time_jump
        t_interval = [t_interval_id, t_inner_bound.strftime('%d/%m/%Y %H:%M'),
                                     t_outer_bound.strftime('%d/%m/%Y %H:%M')]
        vals.append(pd.Series(t_interval))
        t_interval_id += 1

    df_out = pd.DataFrame(vals)
    df_out = df_out.rename(columns={0:'t_interval_id', 1:'t_lower', 2:'t_upper'})
    df_out['t_lower'] = pd.to_datetime(df_out['t_lower'], dayfirst=True)
    df_out['t_upper'] = pd.to_datetime(df_out['t_upper'], dayfirst=True)

    return df_out

def massage_data_file(file_path, tinterval, interval_size, duration_upper):

    def round_minutes(dt, direction, resolution):
        new_minute = (dt.minute // resolution + (1 if direction == 'up' else 0)) * resolution
        return dt + timedelta(minutes=new_minute - dt.minute)

    # Read raw data and ensure the date string is of datetime type
    df_raw = pd.read_csv(file_path, index_col=0)
    df_raw['End Date'] = pd.to_datetime(df_raw['End Date'], dayfirst=True)
    df_raw['Start Date'] = pd.to_datetime(df_raw['Start Date'], dayfirst=True)

    # Discard rental events of extremely long duration
    df_raw = df_raw.loc[df_raw['Duration'] < duration_upper]

    # Count station-to-station connections
    df_graph_weight = df_raw.groupby(['StartStation Id', 'EndStation Id']).size()

    # Map time of bike event to a lower bound of the defined time bins
    df_raw['End Date Lower Bound'] = df_raw['End Date'].apply(round_minutes, **{'direction': 'down', 'resolution': interval_size})
    df_raw['Start Date Lower Bound'] = df_raw['Start Date'].apply(round_minutes, **{'direction': 'down', 'resolution': interval_size})

    # Associate the lower bound of time bin to time interval id and count non-zero occurrences of station and
    # time bin id for both bike arrivals and bike departures
    xx = pd.merge(df_raw, tinterval, left_on='End Date Lower Bound', right_on='t_lower').drop(columns=['t_lower', 't_upper'])
    xx = xx.rename({'t_interval_id' : 'End Date ID'}, axis=1)
    xx = pd.merge(xx, tinterval, left_on='Start Date Lower Bound', right_on='t_lower').drop(columns=['t_lower', 't_upper'])
    xx = xx.rename({'t_interval_id' : 'Start Date ID'}, axis=1)
    xx = xx.drop(columns=['Bike Id', 'Duration',
                          'End Date', 'EndStation Name',
                          'Start Date', 'StartStation Name',
                          'End Date Lower Bound', 'Start Date Lower Bound'])
    g_arrival = xx.groupby(by=['EndStation Id', 'End Date ID'])
    df_arrival = g_arrival.size()
    g_departure = xx.groupby(by=['StartStation Id', 'Start Date ID'])
    df_departure = g_departure.size()

    # Insert zero occurrences
    max_ind_1 = df_arrival.index.max()
    max_ind_2 = df_departure.index.max()
    max_station = int(max(max_ind_1[0], max_ind_2[0]))
    max_timeid = int(max(max_ind_1[1], max_ind_2[1]))
    min_ind_1 = df_arrival.index.min()
    min_ind_2 = df_departure.index.min()
    min_station = int(min(min_ind_1[0], min_ind_2[0]))
    min_timeid = int(min(min_ind_1[1], min_ind_2[1]))
    new_index = pd.MultiIndex.from_product([list(range(min_station, max_station + 1)),
                                            list(range(min_timeid, max_timeid + 1))],
                                           names=['station_id', 'time_id'])
    df_arrival = pd.DataFrame(df_arrival.reindex(new_index, fill_value=0), columns=['arrivals'])
    df_departure = pd.DataFrame(df_departure.reindex(new_index, fill_value=0), columns=['departures'])

    df_all = df_arrival.join(df_departure, how='outer')

    return df_all, df_graph_weight

def merge_data(file_name_root, max_file_index):

    drop_indeces = None
    file_counter = 0
    for ind in range(max_file_index):
        df = pd.read_csv('{}_{}.csv'.format(file_name_root, ind), index_col=(0, 1))
        for ind_ in range(ind + 1, max_file_index):
            print (ind, ind_)
            df_ = pd.read_csv('{}_{}.csv'.format(file_name_root, ind_), index_col=(0, 1))

            ss = df.join(df_, how='inner', lsuffix='_0', rsuffix='_1')
            if len(ss) > 0:
                ss['arrivals'] = ss['arrivals_0'] + ss['arrivals_1']
                ss['departures'] = ss['departures_0'] + ss['departures_1']
                ss = ss.drop(['arrivals_0', 'arrivals_1', 'departures_0', 'departures_1'], axis=1)
                ss.to_csv('data_file_{}.csv'.format(file_counter))
                file_counter += 1

                if drop_indeces is None:
                    drop_indeces = ss.index
                else:
                    drop_indeces = drop_indeces.union(ss.index)

    for ind in range(max_file_index):
        df = pd.read_csv('{}_{}.csv'.format(file_name_root, ind), index_col=(0, 1))
        mask = df.index.isin(drop_indeces)
        df = df[~mask]
        df.to_csv('data_file_{}.csv'.format(file_counter))
        file_counter += 1

def make_graph_weights(graphs):

    cat_graph = pd.concat(graphs)
    cat_graph_all = cat_graph.groupby(['StartStation Id','EndStation Id']).mean()
    max_val = cat_graph_all.max()
    cat_graph_all = (100.0 / max_val) * cat_graph_all
    df = pd.DataFrame(cat_graph_all, columns=['weight'])

    return df

def main(raw_data_file_paths, time_interval_size, lower_t_bound, upper_t_bound, duration_upper):

    # Create a time interval id mapping to real-world times
    tinterval = make_time_interval(lower_t_bound, upper_t_bound, time_interval_size)
    tinterval.to_csv('tinterval.csv', index=False)

    # Reformat a collection of raw data files
    graphs = []
    for k, file_path in enumerate(raw_data_file_paths):
        print (file_path)
        spatiotemp_, graph_weights = massage_data_file(file_path, tinterval, time_interval_size, duration_upper)
        spatiotemp_.to_csv('data_reformat_{}.csv'.format(k))
        graphs.append(graph_weights)

    # Put together the graph weight data
    df_graph_all = make_graph_weights(graphs)
    df_graph_all.to_csv('graph_weight.csv')

    # Deal with duplicate indices in adjacent files
    merge_data('data_reformat', k + 1)

if __name__ == '__main__':

    # File paths to raw data files
    FPS2015 = ['9a-Journey-Data-Extract-23Aug15-05Sep15.csv', '11b Journey Data Extract 01Nov15-14Nov15.csv',
     '5a.JourneyDataExtract03May15-16May15.csv', '3a.JourneyDataExtract01Mar15-15Mar15.csv',
     '10b Journey Data Extract 04Oct15-17Oct15.csv', '9b-Journey-Data-Extract-06Sep15-19Sep15.csv',
     '4a.JourneyDataExtract01Apr15-16Apr15.csv', '2a.JourneyDataExtract01Feb15-14Feb15.csv',
     '7b.JourneyDataExtract12Jul15-25Jul15.csv', '11a Journey Data Extract 18Oct15-31Oct15.csv',
     '6bJourneyDataExtract13Jun15-27Jun15.csv', '6aJourneyDataExtract31May15-12Jun15.csv',
     '8aJourneyDataExtract26Jul15-07Aug15.csv', '1b.JourneyDataExtract18Jan15-31Jan15.csv',
     '12a Journey Data Extract 15Nov15-27Nov15.csv', '3b.JourneyDataExtract16Mar15-31Mar15.csv',
     '7a.JourneyDataExtract28Jun15-11Jul15.csv', '1a.JourneyDataExtract04Jan15-17Jan15.csv',
     '8bJourneyData Extract 08Aug15-22Aug15.csv', '12b Journey Data Extract 28Nov15-12Dec15.csv',
     '4b.JourneyDataExtract 17Apr15-02May15.csv', '2b.JourneyDataExtract15Feb15-28Feb15.csv',
     '13b Journey Data Extract 25Dec15-09Jan16.csv', '13a Journey Data Extract 13Dec15-24Dec15.csv',
     '5b.JourneyDataExtract17May15-30May15.csv', '10a Journey Data Extract 20Sep15-03Oct15.csv']
    fps = ['/Users/andersohrn/Development/london_bike_forecast/data/2015TripDataZip/{}'.format(fp) for fp in FPS2015]

    # Number of minutes of a time interval
    INTERVAL = 60

    # Lower and upper bounds of all relevant times as strings YYYY/MM/DD
    LOWER = '2015/01/01'
    UPPER = '2016/02/01'

    # Highest allowed duration of rental event to be included
    DURATION_UPPER=30000

    # Execute
    main(fps, INTERVAL, LOWER, UPPER, DURATION_UPPER)
