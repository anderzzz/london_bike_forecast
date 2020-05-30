'''Functions to parse raw data files as retrieved from the database of Transport for London. The raw data
is available at https://cycling.data.tfl.gov.uk

The functions extract key information, performs filtering for basic sanity of data points, and creates new
files in a simpler format for parsing by a data loader or other method.

Written by: Anders Ohrn, May 2020

'''
import os
import pandas as pd
from datetime import datetime, timedelta

def make_time_interval(start_date, end_date, interval_size):
    '''Assign sorted integer index to each time interval of defined size between a starting date and
    an ending date.

    Args:
        start_date (str) : Start date in format YYYY/MM/DD
        end_date (str): End date in format YYYY/MM/DD
        interval_size (int): Number of minutes in a time interval

    '''
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

def massage_data_file(file_path, tinterval, interval_size, duration_upper, station_exclude, station_const_ids):
    '''Read raw data file and extract the salient data and format it.

    Args:
        file_path (str): Path to raw data file of CSV format as available in the Transport for London database
        tinterval (DataFrame): Map between integer time index and an interval of date and time
        interval_size (int): Number of minutes in a time interval
        duration_upper (int): Number of seconds above which a rental event is deemed invalid.
        station_exclude (list): List of station IDs for stations to be excluded from all consideration.
        station_const_ids (set): The known station IDs, to use for validation

    '''

    def round_minutes(dt, direction, resolution):
        new_minute = (dt.minute // resolution + (1 if direction == 'up' else 0)) * resolution
        return dt + timedelta(minutes=new_minute - dt.minute)

    # Read raw data and ensure the date string is of datetime type
    df_raw = pd.read_csv(file_path, index_col=0)
    df_raw['End Date'] = pd.to_datetime(df_raw['End Date'], dayfirst=True)
    df_raw['Start Date'] = pd.to_datetime(df_raw['Start Date'], dayfirst=True)

    # Create the header mapper. Truly annoying feature of data is that headers are not named exactly the same over the years
    df_raw = df_raw.rename(columns={'End Station Id' : 'EndStation Id', 'Start Station Id' : 'StartStation Id',
                                    'Duration_Seconds' : 'Duration', 'End Station Name' : 'EndStation Name',
                                    'Start Station Name' : 'StartStation Name'})

    # Some rentals did not terminate in a station. Discard these.
    df_raw = df_raw[~df_raw['EndStation Id'].isna()]

    # Force type
    df_raw['EndStation Id'] = df_raw['EndStation Id'].astype(int)
    df_raw['StartStation Id'] = df_raw['StartStation Id'].astype(int)

    # Discard rental events of extremely long duration
    df_raw = df_raw.loc[df_raw['Duration'] < duration_upper]

    # Validate station IDs
    s_diff = set(df_raw['StartStation Id'].to_list()) - station_const_ids
    e_diff = set(df_raw['EndStation Id'].to_list()) - station_const_ids
    if len(s_diff) > 0:
        raise RuntimeError('In file {} unknown start station IDs: {}'.format(file_path, s_diff))
    if len(e_diff) > 0:
        raise RuntimeError('In file {} unknown end station IDs: {}'.format(file_path, e_diff))

    # Exclude stations
    df_raw = df_raw.loc[~df_raw['StartStation Id'].isin(station_exclude)]
    df_raw = df_raw.loc[~df_raw['EndStation Id'].isin(station_exclude)]

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
    if len(xx) == 0:
        raise RuntimeError('Time interval missing indices for file {}'.format(file_path))

    g_arrival = xx.groupby(by=['EndStation Id', 'End Date ID'])
    df_arrival = g_arrival.size()
    g_departure = xx.groupby(by=['StartStation Id', 'Start Date ID'])
    df_departure = g_departure.size()

    # Raw data reports an event at a time and place, but not absence of an even at a time and place. All missing
    # time-place coordinates are set to zero and the data set extended
    s1 = df_arrival.index.get_level_values('EndStation Id').unique()
    s2 = df_departure.index.get_level_values('StartStation Id').unique()
    s_index = s1.union(s2)
    t1 = df_arrival.index.get_level_values('End Date ID').min()
    t2 = df_departure.index.get_level_values('Start Date ID').max()
    t_index = pd.Int64Index(list(range(t1, t2 + 1)), dtype='int64')
    new_index = pd.MultiIndex.from_product([s_index, t_index],
                                           names=['station_id', 'time_id'])
    df_arrival = pd.DataFrame(df_arrival.reindex(new_index, fill_value=0), columns=['arrivals'])
    df_departure = pd.DataFrame(df_departure.reindex(new_index, fill_value=0), columns=['departures'])

    df_all = df_arrival.join(df_departure, how='outer')

    return df_all, df_graph_weight

def merge_data(file_name_root, max_file_index, file_name_save='data_file'):
    '''The raw data is split between files, which implies that a place-time coordinate on the boundary between
    two time-adjacent files can appear in two files, in one as arrival in the other as departure. This minority of
    coordinates are identified and put into separate file and removed from the main files

    This function assumes all DataFrames can be put in memory. If that is false slight modification needed.

    Args:
        file_name_root (str): File name root of the temporary data files
        max_file_index (int): The highest index of the temporary data files
        file_name_save_root (str, optional): File name root of the disjoint data files

    '''
    drop_indeces = None
    df_all = []

    # Triangular iteration through pairs of files to determine if there are overlapping time-place coordinates
    for ind in range(max_file_index):
        df = pd.read_csv('{}_{}.csv'.format(file_name_root, ind), index_col=(0, 1))
        for ind_ in range(ind + 1, max_file_index):
            print ('Processing file pair {}-{} of {}'.format(ind, ind_, max_file_index))
            df_ = pd.read_csv('{}_{}.csv'.format(file_name_root, ind_), index_col=(0, 1))
            ss = df.join(df_, how='inner', lsuffix='_0', rsuffix='_1')

            # If time-place coordinate overlap found put coordinates in new common file
            if len(ss) > 0:
                ss['arrivals'] = ss['arrivals_0'] + ss['arrivals_1']
                ss['departures'] = ss['departures_0'] + ss['departures_1']
                ss = ss.drop(['arrivals_0', 'arrivals_1', 'departures_0', 'departures_1'], axis=1)
                df_all.append(ss)

                if drop_indeces is None:
                    drop_indeces = ss.index
                else:
                    drop_indeces = drop_indeces.union(ss.index)

    # Discard overlapping time-place coordinates from the main files and
    # save the disjoint files with new name and indexing
    for ind in range(max_file_index):
        df = pd.read_csv('{}_{}.csv'.format(file_name_root, ind), index_col=(0, 1))

        if not drop_indeces is None:
            df = df[~df.index.isin(drop_indeces)]
        df_all.append(df)

    df_all = pd.concat(df_all)

    df_all = add_zeros(df_all, 0.98)
    df_all.to_csv(file_name_save, index=False)

    return list(df_all['station_id'].unique())

def add_zeros(df, station_occ_min_frac=0.0):
    '''On rare occassions stations are off-line for a week or more. The long stretches of missing data is absent,
    but can through this function be replaced with zeros. However, stations that lack data for very long stretches
    should be removed entirely instead.

    Args:
        df (DataFrame): station and time, values of departures and arrivals
        station_occ_min_frac (float) : the fraction of times at which the station has to be present in order
            to be included at all. For all stations above the threshold, missing stretches of data are replaced with zeros.

    Returns:
        df_all (DataFrame): station and time, values of departures and arrivals, with possibly some rarely used
            stations removed compared to input, and some station-time rows added with departures and arrivals set to zero

    '''
    station_occ = df.groupby('station_id').count()
    station_occ_max = station_occ['departures'].max()
    station_occ_min = station_occ_max * station_occ_min_frac
    station_id_above = station_occ.loc[station_occ['departures'] >= station_occ_min].index
    df = df.loc[station_id_above]

    time_id_all = df.groupby('time_id').count().index
    mind_full = pd.MultiIndex.from_product([station_id_above, time_id_all])

    df = df.reindex(mind_full, fill_value=0)
    df = df.reset_index().sort_values(by=['time_id','station_id'])

    return df

def make_graph_weights(graphs, stations_include, norm='median', kwargs_norm={}):
    '''Compute average edge weights for a plurality of weighted and directed graphs

    Args:
        graphs (list): Collection of Series indexed on pairs of station Id with non-zero occurrence in associated raw data
        stations_include (list) : List of stations that are to be included
        norm (str, numeric, optional): How to normalize weights. If string it should be a method of a Pandas
                                       DataFrame that returns a single numeric value. If a numeric value, it is used
                                       to divide all values with.
        kwargs_norm (dict, optional): Arguments to the normalizing method

    '''
    # Create index that contains all station pairs that ever appears in the raw data files
    # and apply this index to all graphs. This is needed to handle rare connections that may be absent in some raw data files
    mega_index = graphs[0].index
    for graph in graphs[1:]:
        mega_index = mega_index.union(graph.index)
    g_expand = [g.reindex(mega_index, fill_value=0) for g in graphs]
    cat_graph = pd.concat(g_expand).reset_index()

    cat_graph = cat_graph.loc[cat_graph['StartStation Id'].isin(stations_include)]
    cat_graph = cat_graph.loc[cat_graph['EndStation Id'].isin(stations_include)]

    cat_graph_all = cat_graph.groupby(['StartStation Id','EndStation Id']).sum()
    cat_graph_all = cat_graph_all.rename(columns={0 : 'total'})

    cat_graph_all['median_norm'] = cat_graph_all['total'] * (1.0 / cat_graph_all['total'].median())

    all_df = []
    for s_ind, df_slice in cat_graph_all.groupby('StartStation Id'):
        mean_slice = df_slice['total'].sum()
        df_slice.loc[:, 'percent_flow'] = 100.0 * df_slice['total'] / mean_slice
        all_df.append(df_slice)
    cat_graph_all = pd.concat(all_df)

    return cat_graph_all

def main(raw_data_file_paths, time_interval_size, lower_t_bound, upper_t_bound,
         duration_upper, stations_exclude, out_filename, station_filename):

    # Create a time interval id mapping to real-world times
    tinterval = make_time_interval(lower_t_bound, upper_t_bound, time_interval_size)
    tinterval.to_csv('tinterval.csv', index=False)

    # Retrieve station constants
    stations_const = pd.read_csv(station_filename)
    stations_const_ids = set(stations_const['station_id'].to_list())

    # Reformat a collection of raw data files
    graphs = []
    for k, file_path in enumerate(raw_data_file_paths):
        print ('Processing... {}'.format(file_path))
        spatiotemp_, graph_weights = massage_data_file(file_path, tinterval, time_interval_size,
                                                       duration_upper, stations_exclude, stations_const_ids)
        spatiotemp_.to_csv('data_reformat_{}.csv'.format(k))
        graphs.append(graph_weights)

    # Deal with duplicate indices in adjacent files
    stations_include = merge_data('data_reformat', k + 1, out_filename)

    # Put together the graph weight data
    df_graph_all = make_graph_weights(graphs, stations_include)
    df_graph_all.to_csv('graph_weight.csv')

if __name__ == '__main__':

    # File paths to raw data files
    files = os.listdir('/Users/andersohrn/Development/london_bike_forecast/data/recent/')
    fps = ['/Users/andersohrn/Development/london_bike_forecast/data/recent/{}'.format(fp) for fp in files]

    # Number of minutes of a time interval
    INTERVAL = 15

    # Name of data output file
    OUTFILENAME='data.csv'

    # Lower and upper bounds of all relevant times as strings YYYY/MM/DD
    LOWER = '2016/12/01'
    UPPER = '2020/09/01'

    # Highest allowed duration of rental event to be included
    DURATION_UPPER=30000

    # Station exclude list. Unlike exclusion in DataSet class, exclusion here removes the stations from the graph
    # and all bike rental events involving the station, not just the former.
    STATION_EXCLUDE = []

    # Station file constants for validation
    STATION_CONST_FILE='/Users/andersohrn/Development/london_bike_forecast/data_reformat_May21/station_id_name.csv'

    # Execute
    main(fps, INTERVAL, LOWER, UPPER, DURATION_UPPER, STATION_EXCLUDE, OUTFILENAME, STATION_CONST_FILE)
