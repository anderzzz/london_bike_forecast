'''Script for training the London bike forecast graph convolutional neural net

Written by: Anders Ohrn, May 2020

'''
from datetime import datetime
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from multichannel_spatiotemporal import STGCN
from london_dataset import LondonBikeDataset

def main(max_epochs, path_model_save, path_model_load,
         dataset_kwargs, dataloader_kwargs, model_kwargs, optimizer_kwargs):
    '''
    Main function to train model on a dataset. Saves the model state to disk as it progresses.

    Args:
        max_epochs (int): Maximum number of epochs in the training
        path_model_save (str): Path to save model. Include up to filename prefix.
        path_model_load (str): Path to model to load state from. If None, state is randomly initialized.
        dataset_kwargs (dict): Keyworded arguments for the creation of a LondonBikeDataset instance
        dataloader_kwargs (dict): Keyworded arguments for the creation of a PyTorch geometric DataLoader instance
        model_kwargs (dict): Keyworded arguments for the spatio-temporal graph convolutional net model instance
        optimizer_kwargs (dict): Keyworded arguments for the SGD optimizer used for the training

    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare the data set and data loader. Dataset preparation can take time and with care taken, the
    # process set to False will reuse already processed files. The meaning of the processed files are
    # currently a manual process, which I could automate later, so use process=True unless you are
    # really sure what you are doing
    london_bike_data = LondonBikeDataset(process=True,
                                         **dataset_kwargs)
    london_bike_loader = DataLoader(london_bike_data, **dataloader_kwargs)

    # Initialize model and optimizer
    model = STGCN(n_spatial_dim=london_bike_data[0].num_nodes, **model_kwargs).to(device)
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)

    # If a prior trained state is available, populate model and optimizer with saved states
    if not path_model_load is None:
        state = torch.load(path_model_load)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

    print ('Start training at {}'.format(datetime.now().isoformat()))

    model.train()
    for epoch in range(max_epochs):
        print ('Epoch: {}'.format(epoch))

        k_batch = 0
        for local_batch in london_bike_loader:
            local_batch = local_batch.to(device)
            print (local_batch.x.shape)

            optimizer.zero_grad()
            out = model(local_batch)
            print ('model for batch {} at {}'.format(k_batch, datetime.now().isoformat()))
            loss = F.mse_loss(out.y, local_batch.y)
            print ('loss data {} at {}'.format(loss, datetime.now().isoformat()))
            loss.backward()
            print ('loss backward at {}'.format(datetime.now().isoformat()))
            optimizer.step()
            print ('optimizer step at {}'.format(datetime.now().isoformat()))

            k_batch += 1

        # This saves model and optimizer states rather than the entire model. This is recommended
        # in Pytorch documentation. Requires loading to take this into account too
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},
            path_model_save + '_{}.tar'.format(epoch))


if __name__ == '__main__':

    #keep_stations = [14, 717, 713, 695, 641]
    #exclude_stations = [x for x in range(1000) if not x in keep_stations]
    exclude_stations = [413, 782, 497, 783, 781, 33, 500, 283, 434, 241, 780, 525, 346, 503, 492, 205, 206, 259, 282,
                        779, 491, 31, 183, 502, 736, 80, 226, 495, 478, 694, 302, 195, 84, 555, 581, 437, 233, 762,
                        631, 162, 105, 102, 435, 121, 53, 36, 319, 344]
    dataset_kwargs = {'weight_filter' : 1.0,
                      'time_id_bounds' : (100, 3000),
                      'time_interval' : 30,
                      'station_id_exclusion' : exclude_stations,
                      'years' : [2015],
                      'time_input_number' : 9,
                      'time_forward_pred' : 2,
                      'name_prefix': 'data',
                      'root_dir' : '/Users/andersohrn/PycharmProjects/torch/data_tmp',
                      'source_dir' : '/Users/andersohrn/Development/london_bike_forecast/data_preprocessed'}

    dataloader_kwargs = {'batch_size' : 50, 'shuffle' : True}

    model_kwargs = {'n_temporal_dim' : 9, 'n_input_channels' : 2,
                    'co_temporal' :64, 'co_spatial' :16, 'time_conv_length' : 3}

    optimizer_kwargs = {'lr' : 0.01,
                        'momentum' : 0.9}

    max_epochs = 2

    path_model_save = '/Users/andersohrn/PycharmProjects/torch/model_save/model_save'
    path_model_load = '/Users/andersohrn/PycharmProjects/torch/model_save/model_store_may19.tar'
    main(max_epochs,
         path_model_save, path_model_load,
         dataset_kwargs, dataloader_kwargs, model_kwargs, optimizer_kwargs)