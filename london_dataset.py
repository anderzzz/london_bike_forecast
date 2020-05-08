import os.path as osp
from os import listdir
import pandas as pd

import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data

TIME_INTERVAL_FILENAME = 'tinterval.csv'
GRAPH_WEIGHT_FILENAME = 'graph_weight.csv'
STATION_ID_FILENAME = 'station_id.csv'
DATA_NAME_ROOT = 'data_file_N.csv'
YEARS = [2015]
BIKE_G_ADJACENCY = 'bike_graph_adjacency_coo.pt'
BIKE_G_WEIGHT = 'bike_graph_weight_coo.pt'

N_STATIONS = 783

data_file_name = lambda n: DATA_NAME_ROOT.replace('N', str(n))

class LondonBikeDataset(Dataset):

    def __init__(self, source_dir, root_dir, toc_file='toc.csv', transform=None, pre_transform=None,
                 station_id_exclusion=None, weight_filter=None, time_interval=30, time_input_number=9,
                 time_forward_pred=1, years=None):

        self.toc = pd.read_csv(source_dir + '/' + toc_file)
        self.source_dir = source_dir

        self.time_interval = time_interval
        self.time_input_number = time_input_number
        self.time_forward_pred = time_forward_pred

        if years is None:
            self.years = YEARS
        else:
            self.years = years
            if not self.years in YEARS:
                raise ValueError('Specified years {} not subset of available years {}'.format(years, YEARS))
        self.station_id_exclusion = station_id_exclusion
        self.weight_filter = weight_filter

        super(LondonBikeDataset, self).__init__(root_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        ret = []
        for year in self.years:
            item = self.toc.loc[(self.toc['year'] == year) & (self.toc['timeinterval'] == self.time_interval)]['dirname'].item()
            data_files = listdir(self.source_dir + '/' + item)
            data_files.remove(TIME_INTERVAL_FILENAME)
            data_files = ['{}/{}/{}'.format(self.source_dir, item, fname) for fname in data_files]
            ret.extend(data_files)
        return ret

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def process(self):

        edge_index, edge_attr = self._make_edges()

        for raw_file_name in self.raw_file_names:
            df = pd.read_csv(raw_file_name)
            if not self.station_id_exclusion is None:
                df = df.loc[~df['station_id'].isin(self.station_id_exclusion)]

            time_id_start = df['time_id'].min()
            time_id_end = df['time_id'].max() - self.time_input_number - self.time_forward_pred

            data_graphs_xy = self._make_time_windows(df, time_id_start, time_id_end, edge_index, edge_attr)

            # PICKLE AND SAVE IN ROOT
            
            raise RuntimeError

    def _make_time_windows(self, df, t_start, t_end, edge_index, edge_attr):

        ret = []
        for t_val in range(t_start, t_end):
            df_x = df.loc[df['time_id'] < t_val + self.time_input_number]
            df_y = df.loc[df['time_id'] == t_val + self.time_input_number + self.time_forward_pred - 1]

            data_xt = []
            for station_id, chunk in df_x.groupby('station_id'):
                data_xt.append([chunk['arrivals'].tolist(),
                              chunk['departures'].tolist()])
            data_yt = [df_y['arrivals'].tolist(), df_y['departures'].tolist()]

            data_graph = Data(x=data_xt, y=data_yt, edge_index=edge_index, edge_attr=edge_attr)
            ret.append(data_graph)

        return ret

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
        self.node_id_station_id_map = dict((n, sid) for n, sid in enumerate(all_ids))
        self.station_id_node_id_map = {v: k for k, v in self.node_id_station_id_map.items()}

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
        index = torch.tensor([[self.station_id_node_id_map[k] for k in station_0],
                              [self.station_id_node_id_map[k] for k in station_1]], dtype=torch.long)
        attr = torch.tensor(weight_vals, dtype=torch.float)

        return index, attr

def test():

    bike_dataset = LondonBikeDataset('/Users/andersohrn/Development/london_bike_forecast/data_preprocessed',
                                     '/Users/andersohrn/PycharmProjects/torch/data_tmp',
                                     weight_filter=0.20)
    bike_dataset.process()
    bike_dataloader = DataLoader(bike_dataset)

if __name__ == '__main__':
    test()