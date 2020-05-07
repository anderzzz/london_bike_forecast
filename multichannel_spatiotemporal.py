import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.data import Data

from .london_dataset import LondonBikeDataset

# Set a convention on input data tensor axes semantics
IDX_CHANNEL = 1
IDX_SPATIAL = 0
IDX_TEMPORAL = 2

class SpatioTemporal(torch.nn.Module):

    def __init__(self, n_spatial, n_temporal,
                 channel_inputs, channel_outputs):

        super(SpatioTemporal, self).__init__()

        self.n_spatial = n_spatial
        self.n_temporal = n_temporal

        self.channel_inputs = channel_inputs
        self.channel_outputs = channel_outputs

        self.permute_norm = (IDX_CHANNEL, IDX_SPATIAL, IDX_TEMPORAL)

class Time1dConvGLU(SpatioTemporal):
    '''Convolution along time dimension with gated linear unit as output activation

    Args:
        channel_inputs (int): Number of channels of input per node
        channel_outputs (int): Number of channels of output per node
        time_convolution_length (int): The kernel size of the time convolution window
        dim_node (int, Optional): The dimension for the nodes on the data. Default is 0
        dim_channels_input (int, Optional): The dimension for the input channels on the data. Default is 1
        dim_timestep (int, Optional): The dimension for the time steps on the data. Default is 2

    '''
    def __init__(self, n_spatial, n_temporal, channel_inputs, channel_outputs,
                 time_convolution_length):

        super(Time1dConvGLU, self).__init__(n_spatial, n_temporal, channel_inputs, channel_outputs)

        self.one_d_conv_1 = torch.nn.Conv1d(in_channels=self.channel_inputs,
                                            out_channels=2 * self.channel_outputs,
                                            kernel_size=time_convolution_length,
                                            groups=1)

        n_temporal_out = self.n_temporal - time_convolution_length + 1
        assert n_temporal_out > 0
        self.layer_norm = torch.nn.LayerNorm([self.n_spatial, n_temporal_out])

    def forward(self, x):
        '''Forward operation'''

        total = []
        for k_node, tensor_node in enumerate(x.split(1, dim=IDX_SPATIAL)):
            pq_out = self.one_d_conv_1(tensor_node)
            y_conv_and_gated_out = F.glu(pq_out, dim=1)
            total.append(y_conv_and_gated_out)

        nodes_times_nonnorm = torch.cat(total, dim=IDX_SPATIAL)
        nodes_times_norm = self.layer_norm(nodes_times_nonnorm.permute(self.permute_norm)).permute(self.permute_norm)

        return nodes_times_norm

class SpatialGraphConv(SpatioTemporal):

    def __init__(self, n_spatial, n_temporal, channel_inputs, channel_outputs):

        super(SpatialGraphConv, self).__init__(n_spatial, n_temporal, channel_inputs, channel_outputs)

        self.gcn = GCNConv(self.channel_inputs, self.channel_outputs)
        self.layer_norm = torch.nn.LayerNorm([self.n_spatial, self.n_temporal])

    def forward(self, data):

        assert self.channel_inputs == data.x.shape[IDX_CHANNEL]

        x_s = data.x
        total = []
        for k_t in range(data.x.shape[IDX_TEMPORAL]):
            x_t = x_s[:,:,k_t]
            y_t = self.gcn(x_t, data.edge_index, data.edge_attr)
            y_t = F.relu(y_t)
            total.append(y_t)

        nodes_times_nonnorm = torch.stack(total, dim=IDX_TEMPORAL)
        nodes_times_norm = self.layer_norm(nodes_times_nonnorm.permute(self.permute_norm)).permute(self.permute_norm)

        return nodes_times_norm

def construct_data():

    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 2, 4],
                               [1, 0, 2, 1, 3, 2, 4, 2]], dtype=torch.long)
    edge_weight = None
    x = torch.tensor([[[1,1,4,1,6,2,3,4,5],[20,20,70,10,70,80,20,10,20]],
                      [[2,3,0,1,0,3,2,6,0],[50,50,60,10,10,20,10,10,10]],
                      [[3,2,2,8,9,8,8,9,6],[40,40,30,40,40,50,30,40,30]],
                      [[5,4,3,3,2,9,1,1,1],[50,40,40,50,80,40,50,20,40]],
                      [[9,0,9,2,8,3,2,9,0],[60,70,70,20,20,60,60,60,60]]], dtype=torch.float)
#    x = torch.tensor([[[1, 15], [2, 12], [3, 22], [0, 14], [1, 19]],
#                      [[2, 10], [4, 10], [6, 25], [3, 20], [6, 24]],
#                      [[3, 19], [7, 30], [5, 28], [1, 21], [1, 26]],
#                      [[4, 12], [4, 19], [3, 29], [0, 25], [4, 21]],
#                      [[5, 22], [0, 12], [1, 16], [2, 16], [5, 22]]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    return data

def main():

    data_step_0 = construct_data()
    edge_index = data_step_0.edge_index
    edge_attr = data_step_0.edge_attr
    n_spatial_data = data_step_0.x.shape[IDX_SPATIAL]
    n_temporal_data = data_step_0.x.shape[IDX_TEMPORAL]
    n_channel_data = data_step_0.x.shape[IDX_CHANNEL]

    c_t_1a, c_s_1, c_t_1b = 64, 16, 64
    c_t_2a, c_s_2, c_t_2b = 64, 16, 64
    time_conv_length = 3

    assert n_temporal_data - 4 * time_conv_length + 4 == 1

    model_t_1a = Time1dConvGLU(n_spatial_data, n_temporal_data,
                               channel_inputs=n_channel_data,
                               channel_outputs=c_t_1a,
                               time_convolution_length=time_conv_length)
    model_s_1 = SpatialGraphConv(n_spatial_data, n_temporal_data - time_conv_length + 1,
                                 channel_inputs=c_t_1a,
                                 channel_outputs=c_s_1)
    model_t_1b = Time1dConvGLU(n_spatial_data, n_temporal_data - time_conv_length + 1,
                               channel_inputs=c_s_1,
                               channel_outputs=c_t_1b,
                               time_convolution_length=time_conv_length)

    model_t_2a = Time1dConvGLU(n_spatial_data, n_temporal_data - 2 * time_conv_length + 2,
                               channel_inputs=c_t_1b,
                               channel_outputs=c_t_2a,
                               time_convolution_length=time_conv_length)
    model_s_2 = SpatialGraphConv(n_spatial_data, n_temporal_data - 3 * time_conv_length + 3,
                                 channel_inputs=c_t_2a,
                                 channel_outputs=c_s_2)
    model_t_2b = Time1dConvGLU(n_spatial_data, n_temporal_data - 3 * time_conv_length + 3,
                               channel_inputs=c_s_2,
                               channel_outputs=c_t_2b,
                               time_convolution_length=time_conv_length)

    model_output = torch.nn.Sequential(torch.nn.Linear(in_features=n_spatial_data * c_t_2b,
                                                       out_features=n_spatial_data * c_t_2b),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(in_features=n_spatial_data * c_t_2b,
                                                       out_features=n_spatial_data * n_channel_data))

    data_step_0_x = data_step_0.x
    data_step_1_x = model_t_1a(data_step_0_x)
    data_step_1 = Data(data_step_1_x, edge_index, edge_attr)
    data_step_2_x = model_s_1(data_step_1)
    data_step_3_x = model_t_1b(data_step_2_x)

    data_step_4_x = model_t_2a(data_step_3_x)
    data_step_4 = Data(data_step_4_x, edge_index, edge_attr)
    data_step_5_x = model_s_2(data_step_4)
    data_step_6_x = model_t_2b(data_step_5_x)

    data_output_x = model_output(data_step_6_x.reshape(n_spatial_data * c_t_2b))
    data_output_x = data_output_x.reshape(n_spatial_data, n_channel_data)
    data_output = Data(data_output_x, edge_index, edge_attr)

#    print (data_output.x)
    print (data_output.x.shape)

if __name__ == '__main__':
    main()