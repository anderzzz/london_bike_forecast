'''Script for training the London bike forecast graph convolutional neural net

Written by: Anders Ohrn, May 2020

'''
from datetime import datetime

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from multichannel_spatiotemporal import STGCN
from london_dataset import LondonBikeDataset
from consts import EXCLUDE_STATIONS

def main(max_epochs, path_model_save, path_model_load, update_saved_model_freq,
         dataset_kwargs, dataloader_kwargs, model_kwargs, optimizer_kwargs):
    '''
    Main function to train model on a dataset. Saves the model state to disk as it progresses.

    Args:
        max_epochs (int): Maximum number of epochs in the training
        path_model_save (str): Path to save model. Include up to filename prefix, e.g. `/my_computer/my_dir/file`
        path_model_load (str): Path to model to load state from. If None, state is randomly initialized.
        update_saved_model_freq (int): How many optimization steps between updates to the saved model.
        dataset_kwargs (dict): Keyworded arguments for the creation of a LondonBikeDataset instance
        dataloader_kwargs (dict): Keyworded arguments for the creation of a PyTorch geometric DataLoader instance
        model_kwargs (dict): Keyworded arguments for the spatio-temporal graph convolutional net model instance
        optimizer_kwargs (dict): Keyworded arguments for the SGD optimizer used for the training

    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(path_model_save + '_input_params.txt', 'w') as fout:
        print ('Training start:{}\n'.format(datetime.now().isoformat()), file=fout)
        print ('dataset kwargs:\n {}\n'.format(dataset_kwargs), file=fout)
        print ('dataloader kwargs:\n {}\n'.format(dataloader_kwargs), file=fout)
        print ('model kwargs:\n {}\n'.format(model_kwargs), file=fout)
        print ('optimizer kwargs:\n {}'.format(optimizer_kwargs), file=fout)

    london_bike_data = LondonBikeDataset(**dataset_kwargs)
    london_bike_loader = DataLoader(london_bike_data, **dataloader_kwargs)
    if 'create_from_source' in dataset_kwargs:
        if dataset_kwargs['create_from_source']:
            london_bike_data.write_creation_params()

    model = STGCN(n_spatial_dim=london_bike_data[0].num_nodes, **model_kwargs).to(device)
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)

    if not path_model_load is None:
        state = torch.load(path_model_load)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

    model.train()
    for epoch in range(max_epochs):
        print ('Epoch {} at {}'.format(epoch, datetime.now().isoformat()))

        for k_batch, local_batch in enumerate(london_bike_loader):
            local_batch = local_batch.to(device)
            optimizer.zero_grad()

            print ('Shape of batch {}: {}'. format(k_batch, local_batch.x.shape))
            out = model(local_batch)
            print ('...model evaluated at {}'.format(datetime.now().isoformat()))
            loss = F.mse_loss(out.y, local_batch.y)
            print ('...with loss {} at {}'.format(loss, datetime.now().isoformat()))
            loss.backward()
            print ('...gradients done at {}'.format(datetime.now().isoformat()))
            optimizer.step()
            print ('...optimization step done at {}'.format(datetime.now().isoformat()))

            if k_batch % update_saved_model_freq == 0:
                # Save model and optimizer states rather than the entire model.
                torch.save({'epoch': epoch,
                            'k_batch': k_batch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss},
                            path_model_save + '.tar')


if __name__ == '__main__':

    dataset_kwargs = {'lower_weight' : 1.0,
                      'common_weight' : 1.0,
                      'time_id_bounds' : (2976, 38015),
                      'time_shuffle' : True,
                      'sample_size' : 960,
                      'create_from_source' : True,
                      'station_exclusion' : EXCLUDE_STATIONS,
                      'ntimes_leading' : 9,
                      'ntimes_forward' : 1,
                      'source_graph_file' : 'graph_weight.csv',
                      'source_data_files' : 'dataraw_15m.csv',
                      'root_dir' : '/Users/andersohrn/PycharmProjects/torch/data_tmp',
                      'source_dir' : '/Users/andersohrn/Development/london_bike_forecast/data_reformat_May21/1701_2004_15m'}

    dataloader_kwargs = {'batch_size' : 64, 'shuffle' : True}

    model_kwargs = {'n_temporal_dim' : 9, 'n_input_channels' : 2,
                    'co_temporal' :64, 'co_spatial' :16, 'time_conv_length' : 3}

    optimizer_kwargs = {'lr' : 0.01,
                        'momentum' : 0.9}

    max_epochs = 2

    path_model_save = '/Users/andersohrn/PycharmProjects/torch/model_save/model_save'
    #path_model_load = '/Users/andersohrn/PycharmProjects/torch/model_save/model_store_may24_1.tar'
    path_model_load = None
    main(max_epochs,
         path_model_save, path_model_load, 3,
         dataset_kwargs, dataloader_kwargs, model_kwargs, optimizer_kwargs)