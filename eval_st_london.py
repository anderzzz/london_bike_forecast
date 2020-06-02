'''Script for evaluating the London bike forecast graph convolutional neural net

Written by: Anders Ohrn, May 2020

'''
import pandas as pd
import torch

from torch_geometric.data import DataLoader

from multichannel_spatiotemporal import STGCN
from london_dataset import LondonBikeDataset
from consts import EXCLUDE_STATIONS

def make_viz_stations(model, data_loader, viz_stations, station2id):

    inds = [station2id[k] for k in viz_stations]
    inds_slicer = [[ind * 2, ind *2 + 1] for ind in inds]
    inds_slicer = torch.tensor(inds_slicer).flatten()
    totals_predict = []
    totals_actual = []
    for n_time, data in enumerate(data_loader):
        predict = model(data).y
        actual = data.y
        v_predict = torch.take(predict, inds_slicer)
        v_actual = torch.take(actual, inds_slicer)
        totals_predict.extend(list(v_predict))
        totals_actual.extend(list(v_actual))

    mind = pd.MultiIndex.from_product([['arrivals', 'departures'], range(n_time + 1), viz_stations],
                                      names=['event_type', 'local_time_id', 'station_id'])
    df_1 = pd.DataFrame(totals_actual, index=mind, columns=['actual_count'])
    df_2 = pd.DataFrame(totals_predict, index=mind, columns=['predicted_count'])
    df = df_1.join(df_2)
    print (df)

def main(path_model_load, dataset_kwargs, dataloader_kwargs, model_kwargs,
         viz_stations=None):
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
    london_bike_data = LondonBikeDataset(**dataset_kwargs)
    london_bike_loader = DataLoader(london_bike_data, **dataloader_kwargs)

    # Initialize model and load state
    model = STGCN(n_spatial_dim=london_bike_data[0].num_nodes, **model_kwargs).to(device)
    state = torch.load(path_model_load)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    if not viz_stations is None:
        make_viz_stations(model, london_bike_loader, viz_stations, london_bike_data.station_id_2_node_id_map)


if __name__ == '__main__':

    dataset_kwargs = {'root_dir' : '/Users/andersohrn/PycharmProjects/torch/data_tmp',
                      'source_dir' : '/Users/andersohrn/Development/london_bike_forecast/data_reformat_May21/1701_2004_15m',
                      'source_data_files' : 'dataraw_15m.csv',
                      'source_graph_file' : 'graph_weight.csv',
                      'weight_type' : 'percent_flow',
                      'common_weight' : 1.0,
                      'lower_weight' : 1.0,
                      'time_shuffle' : False,
                      'create_from_source' : True,
                      'ntimes_leading' : 9,
                      'ntimes_forward' : 4,
                      'station_exclusion' : EXCLUDE_STATIONS,
                      'time_id_bounds' : (5951, 6047)}

    dataloader_kwargs = {'batch_size' : 1, 'shuffle' : False}

    model_kwargs = {'n_temporal_dim' : 9, 'n_input_channels' : 2,
                    'co_temporal' :64, 'co_spatial' :16, 'time_conv_length' : 3}

    main(path_model_load='/Users/andersohrn/PycharmProjects/torch/model_save_15min_4forward_1percent/model_save.tar',
         dataset_kwargs=dataset_kwargs, dataloader_kwargs=dataloader_kwargs, model_kwargs=model_kwargs,
         viz_stations=[14, 34, 593])