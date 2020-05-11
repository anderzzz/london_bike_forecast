import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from multichannel_spatiotemporal import STGCN
from london_dataset import LondonBikeDataset

PATH_RAW = '/Users/andersohrn/Development/london_bike_forecast/data_preprocessed'
PATH_PROCESS = '/Users/andersohrn/PycharmProjects/torch/data_tmp'

def main(max_epochs, dataset_kwargs, dataloader_kwargs, model_kwargs, optimizer_kwargs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    london_bike_data = LondonBikeDataset(PATH_RAW, PATH_PROCESS,
                                         process=True,
                                         **dataset_kwargs)
    london_bike_loader = DataLoader(london_bike_data, **dataloader_kwargs)
    model = STGCN(n_spatial_dim=london_bike_data[0].num_nodes, **model_kwargs).to(device)
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)

    model.train()
    for epoch in range(max_epochs):
        print (epoch)
        for local_batch in london_bike_loader:
            local_batch = local_batch.to(device)
            print (local_batch.y.shape)

            optimizer.zero_grad()
            out = model(local_batch)
            print (out.y.shape)
            print ('a0')
            loss = F.mse_loss(out.y, local_batch.y)
            print ('a1')
            loss.backward()
            print ('a2')
            optimizer.step()
            print ('a3')
            raise RuntimeError


if __name__ == '__main__':

    dataset_kwargs = {'weight_filter' : 1.0,
                      'time_id_bounds' : (100, 200),
                      'time_interval' : 30,
                      'years' : [2015],
                      'time_input_number' : 9,
                      'time_forward_pred' : 6}

    dataloader_kwargs = {'batch_size' : 2}

    model_kwargs = {'n_temporal_dim' : 9, 'n_input_channels' : 2,
                    'co_temporal' :64, 'co_spatial' :16, 'time_conv_length' : 3}

    optimizer_kwargs = {'lr' : 0.1,
                        'momentum' : 0.9}

    max_epochs = 10

    main(max_epochs, dataset_kwargs, dataloader_kwargs, model_kwargs, optimizer_kwargs)