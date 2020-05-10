import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data

import pandas as pd

'''Standard name of time interval index file (input)'''
TIME_INTERVAL_FILENAME = 'tinterval.csv'

'''Standard name of the weighted directed station to station file (input)'''
GRAPH_WEIGHT_FILENAME = 'graph_weight_2015.csv'

'''Standard name of the station ID to name map file (input)'''
STATION_ID_FILENAME = 'station_id.csv'

'''NN'''
TIME_SLICE_NAME = 'time_slice'

'''Standard name of pickled torch tensor of graph adjacency matrix (output)'''
BIKE_G_ADJACENCY = 'bike_graph_adjacency_coo.pt'

'''Standard name of pickled torch tensor of graph weight matrix (output)'''
BIKE_G_WEIGHT = 'bike_graph_weight_coo.pt'

'''Years available for possible processing'''
YEARS = [2015]

'''Time intervals available for possible processing'''
T_INTERVALS = [10, 20, 30, 60]

N_STATIONS = 783

class LondonBikeDataset(Dataset):
    '''The Geometric PyTorch dataset class to make raw data available for Pytorch processing, including DataLoaders

    Args:
        source_dir (str): Path to source directory for input data to process. This should contain data files
        obtained from `rawdata_reformat`, and a table of content file to aid navigation.
        root_dir (str): Path to directory for pickled output files, such that processing step can be run just once.
        toc_file (str, optional): File name of the table of content CSV file in the `source_dir`.
        transform
        pre_transform
        station_id_exclusion (list, optional): If certain stations should be entirely excluded from consideration,
        provide their corresponding station IDs in a list. Default is no exclusion.
        weight_filter (float, optional):
        time_id_bounds (tuple, optional): If only data for subset of times are to be considered, provide the lower
        and upper bound time indeces as a tuple of two integers.
        time_interval (int, optional): Which data to use with respect to the time interval resolution. Default 30 minutes.
        years (list, optional): Which data to use with respect to the years. Default 2015.
        time_input_number(int, optional): How many preceeding time values to provide the predictor with. Must be a value
        compatible with the model architecture. Default 9.
        time_forward_pred(int, optional): How far into the future from the most recent given data point the prediction
        is aimed. Default 1.

    '''
    def __init__(self, source_dir, root_dir, toc_file='toc.csv',
                 transform=None, pre_transform=None,
                 station_id_exclusion=None, weight_filter=None, time_id_bounds=None,
                 time_interval=30, years=[2015],
                 time_input_number=9, time_forward_pred=1):

        # Data input: location
        self.source_dir = source_dir
        self.toc = pd.read_csv(self.source_dir + '/' + toc_file)

        # Data input: year and time resolution subset
        self.time_interval = time_interval
        if not self.time_interval in T_INTERVALS:
            raise ValueError('Specified time interval resolution {} not available'.format(time_interval))
        self.years = years
        if not len(set(self.years) - set(YEARS)) == 0:
            raise ValueError('Specified years {} not subset of available years {}'.format(years, YEARS))

        # Time slice: how many inputs, X, and how far into the future from X the predicted value, y, is
        self.time_input_number = time_input_number
        self.time_forward_pred = time_forward_pred

        # Filters
        self.station_id_exclusion = station_id_exclusion
        self.weight_filter = weight_filter
        self.time_id_bounds = time_id_bounds

        self._processed_file_names = ['qqq']

        super(LondonBikeDataset, self).__init__(root_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data_{}m_{}/data_{}m_{}.csv'.format(self.time_interval, year, self.time_interval, year) for year in self.years]

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def len(self):
        return len(self.processed_file_names)

    def process(self):

        edge_index, edge_attr = self._make_edges()

        for raw_file_name in self.raw_file_names:
            df = pd.read_csv(self.source_dir + '/' + raw_file_name)

            # Apply filters on data to situate in the graph
            if not self.station_id_exclusion is None:
                df = df.loc[~df['station_id'].isin(self.station_id_exclusion)]
            if not self.time_id_bounds is None:
                df = df.loc[(df['time_id'] >= self.time_id_bounds[0]) & \
                            (df['time_id'] <= self.time_id_bounds[1])]

            # The range of sliding window origins for which to produce time-ordered dependent and independent data
            time_id_start = df['time_id'].min()
            time_id_end = df['time_id'].max() - self.time_input_number - self.time_forward_pred

            # Create data graphs for each sliding window and store as processed files
            count = 0
            for data_g_xt, data_g_yt in self._make_time_windows(df, time_id_start, time_id_end):
                data_graph = Data(x=torch.tensor(data_g_xt, dtype=torch.float),
                                  y=torch.tensor(data_g_yt, dtype=torch.float),
                                  edge_index=edge_index, edge_attr=edge_attr)
                torch.save(data_graph, self.root + '/' + TIME_SLICE_NAME + '_{}.pt'.format(count))
                self._processed_file_names.append('{}_{}.pt'.format(TIME_SLICE_NAME, count))
                count += 1

    def get(self, idx):
        data = torch.load(self.root +'/' + TIME_SLICE_NAME + '_{}.pt'.format(idx))
        return data

    def _make_time_windows(self, df, t_start, t_end):

        null_list = [0] * self.time_input_number
        for t_val in range(t_start, t_end):
            df_x = df.loc[(df['time_id'] < t_val + self.time_input_number) & \
                          (df['time_id'] >= t_val)]
            df_y = df.loc[df['time_id'] == t_val + self.time_input_number + self.time_forward_pred - 1]

            # Move from pandas to torch tensor compatible data. Because a given time slice may have been created
            # from raw data that does not include events at all stations in the graph, all stations are accounted
            # for through null initialization, which are replaced if actual station data is available.
            data_xt = [[null_list, null_list]] * len(self.station_id_2_node_id_map)
            for station_id, chunk in df_x.groupby('station_id'):
                ind = self.station_id_2_node_id_map[station_id]
                data_xt[ind] = [chunk['arrivals'].tolist(),
                                chunk['departures'].tolist()]
            data_yt = [[0, 0]] * len(self.station_id_2_node_id_map)
            for station_id, chunk in df_y.groupby('station_id'):
                ind = self.station_id_2_node_id_map[station_id]
                data_yt[ind] = [chunk['arrivals'].item(), chunk['departures'].item()]

            yield data_xt, data_yt

    def _make_edges(self):

        # Obtain edge data and ensure proper types
        df_gw = pd.read_csv('{}/{}'.format(self.source_dir, GRAPH_WEIGHT_FILENAME))
        df_gw['StartStation Id'] = df_gw['StartStation Id'].astype(int)
        df_gw['EndStation Id'] = df_gw['EndStation Id'].astype(int)
        df_gw['weight'] = df_gw['weight'].astype(float)

        # Discard specific stations if exclusion filter provided
        if not self.station_id_exclusion is None:
            df_gw = df_gw.loc[~df_gw['StartStation Id'].isin(self.station_id_exclusion)]
            df_gw = df_gw.loc[~df_gw['EndStation Id'].isin(self.station_id_exclusion)]

        # Create map between station ids and node id.
        all_ids = set(df_gw['StartStation Id'].unique()).union(set(df_gw['EndStation Id'].unique()))
        self.node_id_2_station_id_map = dict((n, sid) for n, sid in enumerate(all_ids))
        self.station_id_2_node_id_map = {v: k for k, v in self.node_id_2_station_id_map.items()}

        # Make graph non-directional with averaged weights and without self-loops
        df1 = df_gw.loc[df_gw['StartStation Id'] != df_gw['EndStation Id']].set_index(['StartStation Id', 'EndStation Id'])
        df2 = df1.reorder_levels(['EndStation Id', 'StartStation Id'])
        df2 = df2.reindex(df1.index, fill_value=0.0)
        df3 = pd.concat([df1, df2], axis=1).mean(axis=1)
        df3 = df3.reset_index()

        # Remove edges on basis of weight filter, if any
        if not self.weight_filter is None:
            df3 = df3.loc[df3[0] >= self.weight_filter]

        # Put data in format expected by Pytorch
        station_0 = df3['StartStation Id'].tolist()
        station_1 = df3['EndStation Id'].tolist()
        weight_vals = df3[0].tolist()
        index = torch.tensor([[self.station_id_2_node_id_map[k] for k in station_0],
                              [self.station_id_2_node_id_map[k] for k in station_1]], dtype=torch.long)
        attr = torch.tensor(weight_vals, dtype=torch.float)

        return index, attr

def test():

    bike_dataset = LondonBikeDataset('/Users/andersohrn/Development/london_bike_forecast/data_preprocessed',
                                     '/Users/andersohrn/PycharmProjects/torch/data_tmp',
                                     weight_filter=0.01,
                                     time_id_bounds=(100,200),
                                     time_forward_pred=6)
    bike_dataset.process()
    bike_dataset.get(0)
    bike_dataset.get(30)
    bike_dataloader = DataLoader(bike_dataset)

if __name__ == '__main__':
    test()