'''Script for evaluating the London bike forecast graph convolutional neural net

Written by: Anders Ohrn, May 2020

'''
import torch

from torch_geometric.data import DataLoader

from multichannel_spatiotemporal import STGCN
from london_dataset import LondonBikeDataset
from consts import EXCLUDE_STATIONS

def main(path_model_load, dataset_kwargs, dataloader_kwargs, model_kwargs):
    '''
    Main function to evaluate model on a dataset.

    Args:
        path_model_load (str): Path to model to load state from. If None, state is randomly initialized.
        dataset_kwargs (dict): Keyworded arguments for the creation of a LondonBikeDataset instance
        dataloader_kwargs (dict): Keyworded arguments for the creation of a PyTorch geometric DataLoader instance
        model_kwargs (dict): Keyworded arguments for the spatio-temporal graph convolutional net model instance

    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare the data set and data loader. Dataset preparation can take time and with care taken, the
    # process set to False will reuse already processed files. The meaning of the processed files are
    # currently a manual process, which I could automate later, so use process=True unless you are
    # really sure what you are doing
    london_bike_data = LondonBikeDataset(process=True,
                                         **dataset_kwargs)
    london_bike_loader = DataLoader(london_bike_data, **dataloader_kwargs)

    # Initialize model and load state
    model = STGCN(n_spatial_dim=london_bike_data[0].num_nodes, **model_kwargs).to(device)
    state = torch.load(path_model_load)
    model.load_state_dict(state['model_state_dict'])

    model.eval()

    for local_batch in london_bike_loader:
        local_batch = local_batch.to(device)
        print(local_batch.x.shape)
        out = model(local_batch)
        print (out)

        raise RuntimeError

if __name__ == '__main__':

    dataset_kwargs = {'weight_filter' : 1.0,
                      'time_id_bounds' : (100, 3000),
                      'time_interval' : 30,
                      'station_id_exclusion' : EXCLUDE_STATIONS,
                      'years' : [2017],
                      'time_input_number' : 9,
                      'time_forward_pred' : 2,
                      'name_prefix': 'data',
                      'root_dir' : '/Users/andersohrn/PycharmProjects/torch/data_tmp_test',
                      'source_dir' : '/Users/andersohrn/Development/london_bike_forecast/data_preprocessed'}

    dataloader_kwargs = {'batch_size' : 50, 'shuffle' : True}

    model_kwargs = {'n_temporal_dim' : 9, 'n_input_channels' : 2,
                    'co_temporal' :64, 'co_spatial' :16, 'time_conv_length' : 3}

    main(FOOBAR)