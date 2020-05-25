'''London bike sharing dataset class for Pytorch geometric. The class builds on the Dataset parent from
Pytorch Geometric.

The raw data can be selected and transformed in several ways in order to create input and output data
situated on the graph. That includes the time interval resolution. However most important is the
way a sliding window is used on the raw data in order to create the input and output. For example, the raw
data contains two data channels per bike station (the spatial dimension) and time interval (the temporal
dimension). In a prediction task, for a given bike station, the values at the two data channels are given
for a certain number of input temporal coordinates (e.g. t3,t2,t1), and during training the value of the
two data channels at one temporal coordinate into the future is given (e.g. tX). The number of inputs and
the coordinate of the output relative the inputs define the sliding window. As that sliding window moves
over the raw data, the input and expected output for the training task are obtained. The precise number
of such data slices depend non-trivially on the inputs to the class. For that reason the class deviates in
the initialization from the template Dataset class.

Written by: Anders Ohrn, May 2020

'''
import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data

from os import listdir, makedirs, remove
import os.path as osp
import pandas as pd

'''File prefix for the processed files'''
TIME_SLICE_NAME = 'time_slice'

class LondonBikeDataset(Dataset):
    '''The Geometric PyTorch dataset class to make raw data available for Pytorch processing, including DataLoaders

    The workflow is as this: (1) Data from the `rawdata_reformat` is taken as input, (2) sliced to create the
    input data and output data for training and stored, such that a DataLoader can construct batches for training.

    The raw data has to exist in a folder hierarchy with paths as the following format:

    `'{}/{}_{}m/{}_{}m.csv'.format(source_dir, dir_name_prefix, time_interval, raw_name_prefix, time_interval)`

    The slicing of the data into input-output pairs for supervised learning can take considerable time (minutes to hours)
    if the range of time indices is wide (1000+). Therefore the time slices are stored, such that if additional
    training is done for the same input data, the `process` can be set to `False` and prior data is used. Note that
    no validation is done to ensure the data slices are consistent with the initialization parameters. Set `process`
    to `False` with extreme caution.

    Args:
        source_dir (str): Path to source directory for input data to process. This folder has to conform to the
            folder hierarchy defined in class documentation. Typically the data content is created by a prior
            execution of `rawdata_reformat`
        root_dir (str): Path to directory where to the pickled pre-processed time slices are stored. If `process`
            is `True` the content in this directory is created anew. If `process` is `False` the content in this
            directory is only referenced. The files in this directory are not meant to be moved, read or adjusted.
        dir_name_prefix (str, optional): Prefix to data directories within the source directory. Default is `bikedata`
        raw_name_prefix (str, optional): Prefix to the data files within the data directories. Default is `dataraw`
        graph_weight_name (str, optional): File name to the weighted graph that is stored in the source directory and
            which applies to all raw data variants. Default is `graph_weight.csv`
        transform : No functionality at the moment.
        pre_transform : No functionality at the moment.
        station_id_exclusion (list, optional): If certain stations should be entirely excluded from consideration,
            provide their station IDs in a list. Default is no exclusion.
        weight_filter (float, optional): If the edges in the station-to-station graph should be pruned on basis
            of weight, provide the lower bound of weight for edges to keep. Default is no pruning.
        time_id_bounds (tuple, optional): If only data for subset of times are to be considered, provide the lower
            and upper bound time indeces as a tuple of two integers. Default is no time slicing.
        time_interval (int, optional): Which data directory to use to create the data set. The data directories are
            differentiated on the number of minutes per time interval. Default is 30.
        time_input_number(int, optional): How many contiguous time intervals of data to provide as input, the "x", to
            the model. Value must be compatible with the model architecture. Default 9.
        time_forward_pred(int, optional): How many time intervals into the future from the most recent given
            time interval given to the prediction that should be predicted by the model, the "y". Default 1.
        process (bool, optional): Flag to determine if the class initialization should create all the time slices or
            only link to already processed data in the `root_dir`. Default is `True`

    '''
    def __init__(self, source_dir, root_dir,
                 dir_name_prefix='bikedata', raw_name_prefix='dataraw', graph_weight_name='graph_weight.csv',
                 transform=None, pre_transform=None,
                 station_id_exclusion=None, weight_filter=None, time_id_bounds=None,
                 time_interval=30,
                 time_input_number=9, time_forward_pred=1,
                 process=True):

        # Paths, prefixes and file names
        self.source_dir = source_dir
        self.root = root_dir
        self.dir_name_prefix = dir_name_prefix
        self.raw_name_prefix = raw_name_prefix
        self.graph_weight_name = graph_weight_name

        # Select subset of raw data on basis of time resolution
        self.time_interval = time_interval

        # Time slice: how many inputs, X, and how far into the future from X the predicted value, y, is
        self.time_input_number = time_input_number
        self.time_forward_pred = time_forward_pred

        # Filters
        self.station_id_exclusion = station_id_exclusion
        self.weight_filter = weight_filter
        self.time_id_bounds = time_id_bounds

        # Either create files or simply link to existing ones. This slightly convoluted way is required because
        # the total number of processed files is only known after the processing is done. Therefore the
        # process command must be made explicit unlike the template Dataset
        if process:
            if not osp.exists(root_dir):
                makedirs(root_dir)
            else:
                for ff in listdir(root_dir):
                    remove(root_dir + '/' + ff)
            self.create_torch_data()
        else:
            self._processed_file_names = listdir(root_dir)

        super(LondonBikeDataset, self).__init__(root_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['{}_{}m/{}_{}m.csv'.format(self.dir_name_prefix, self.time_interval,
                                           self.raw_name_prefix, self.time_interval)]

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.root +'/' + TIME_SLICE_NAME + '_{}.pt'.format(idx))
        return data

    def create_torch_data(self):
        '''Method to construct the time slices given the raw data, and populate the weighted graph,
        and then to write the graph to the root directory from which the DataLoader later takes data

        '''
        edge_index, edge_attr = self._make_edges()

        self._processed_file_names = []
        for raw_file_name in self.raw_file_names:
            df = pd.read_csv(self.source_dir + '/' + raw_file_name)

            # Apply filters on data to situate in the graph
            if not self.station_id_exclusion is None:
                df = df.loc[~df['station_id'].isin(self.station_id_exclusion)]
            if not self.time_id_bounds is None:
                df = df.loc[(df['time_id'] >= self.time_id_bounds[0]) & \
                            (df['time_id'] <= self.time_id_bounds[1])]

            # In the event that filters removed all data, jump to next iteration
            if len(df) == 0:
                continue

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

    def _make_time_windows(self, df, t_start, t_end):

        null_list = [0] * self.time_input_number
        for t_val in range(t_start, t_end):
            df_x = df.loc[(df['time_id'] < t_val + self.time_input_number) & \
                          (df['time_id'] >= t_val)]
            df_y = df.loc[df['time_id'] == t_val + self.time_input_number + self.time_forward_pred - 1]

            if len(df_x) == 0 or len(df_y) == 0:
                raise RuntimeError('Encountered missing time for time_X: {}, and time_Y: {}'.format(t_val, t_val + self.time_input_number + self.time_forward_pred - 1))

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
        df_gw = pd.read_csv('{}/{}'.format(self.source_dir, self.graph_weight_name))
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

    bike_dataset = LondonBikeDataset('/Users/andersohrn/Development/london_bike_forecast/data_reformat_May21',
                                     '/Users/andersohrn/PycharmProjects/torch/data_tmp',
                                     dir_name_prefix='1701_2004',
                                     weight_filter=0.01,
                                     time_id_bounds=(100,200),
                                     time_forward_pred=6,
                                     process=True)
    bike_dataset.get(0)
    bike_dataset.get(30)
    bike_dataloader = DataLoader(bike_dataset, batch_size=4)
    for dd in bike_dataloader:
        print (dd)

if __name__ == '__main__':
    test()