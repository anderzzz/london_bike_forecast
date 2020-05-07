import os.path as osp
import pandas as pd

import torch
from torch_geometric.data import Dataset

class LondonBikeDataset(Dataset):

    def __init__(self, toc_file, root_dir, transform=None, pre_transform=None,
                 weight_filter=None, time_interval=30):

        super(LondonBikeDataset, self).__init__(root_dir, transform, pre_transform)

        self.toc = pd.read_csv(toc_file)

    def process(self):

        # Make adjacency matrix COO on basis of weight filter then load

def test():

    dataset = LondonBikeDataset()

if __name__ == '__main__':
    test()