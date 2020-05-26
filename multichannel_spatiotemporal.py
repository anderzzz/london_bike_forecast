'''The SpatioTemporal Graph Convolutional Net (STGCN) method involving four distinct 1D convolutions along the
temporal dimension and two distinct graph convolutions along the spatial dimensions. The method involves two
layer normalizations and one fully connected layer with ReLu activation

The method is almost identical to the method published by Yu, Yin, and Zhu (2018) arXiv:1709.04875v4

Written by Anders Ohrn, May 2020

'''
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.data import Data

# Set a convention on input data tensor axes semantics
IDX_CHANNEL = 1
IDX_SPATIAL = 0
IDX_TEMPORAL = 2

class SpatioTemporal(torch.nn.Module):
    '''Base Class for the Spatio Temporal classes

    '''
    def __init__(self, n_spatial, n_temporal,
                 channel_inputs, channel_outputs):

        super(SpatioTemporal, self).__init__()

        self.n_spatial = n_spatial
        self.n_temporal = n_temporal

        self.channel_inputs = channel_inputs
        self.channel_outputs = channel_outputs

class Time1dConvGLU(SpatioTemporal):
    '''Convolution along time dimension with gated linear unit as output activation

    The temporal convolution applies a 1D convolution of a set kernel size. The temporal convolution is applied
    to all spatial units in the forward operation. That is, the parameters are the same for all spatial
    units. The output of the 1D convolution is passed through a gated linear unit, hence the channel output is
    cut in half.

    The input dimension is (N, T, Cin) and the output dimension is (N, T-k+1, Cout), where N is the number of
    spatial units, T is the number of temporal units per spatial unit, k is the kernel size, Cin the number of
    channels of the input, and Cout the number of channels of the output.

    Number of parameters scale as: 2 * Cout + 2 * Cout * Cin * k

    Args:
        n_spatial (int): Number of spatial units in mode, equal to number of nodes in graph
        n_temporal (int): Number of contiguous temporal units per spatial unit
        channel_inputs (int): Number of channels of input per node
        channel_outputs (int): Number of channels of output per node
        time_convolution_length (int): The kernel size of the time convolution window

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

    def forward(self, x):
        '''Forward operation. Apply the same temporal convolution to all spatial units'''

        total = []
        for k_node, tensor_node in enumerate(x.split(1, dim=IDX_SPATIAL)):
            pq_out = self.one_d_conv_1(tensor_node)
            y_conv_and_gated_out = F.glu(pq_out, dim=1)
            total.append(y_conv_and_gated_out)

        nodes_spatial = torch.cat(total, dim=IDX_SPATIAL)

        return nodes_spatial

class SpatialGraphConv(SpatioTemporal):
    '''Convolution in spatial dimension with graph convolution

        The graph convolution is applied to all temporal layers, so one set of parameters apply to all
        temporal layers. The graph convolution is based on the graph Laplacian, and hence only spatial units
        that are connected can contribute to each others outputs after convolution.

        The input dimension is (N, T, Cin) and the output dimension is (N, T, Cout), where N is the number of
        spatial units, T is the number of temporal units per spatial unit, Cin the number of
        channels of the input, and Cout the number of channels of the output.

        Args:
            n_spatial (int): Number of spatial units in mode, equal to number of nodes in graph
            n_temporal (int): Number of contiguous temporal units per spatial unit
            channel_inputs (int): Number of channels of input per node
            channel_outputs (int): Number of channels of output per node

    '''
    def __init__(self, n_spatial, n_temporal, channel_inputs, channel_outputs):

        super(SpatialGraphConv, self).__init__(n_spatial, n_temporal, channel_inputs, channel_outputs)

        self.gcn = GCNConv(self.channel_inputs, self.channel_outputs)

    def forward(self, data):

        assert self.channel_inputs == data.x.shape[IDX_CHANNEL]

        x_s = data.x
        total = []
        for k_t in range(data.x.shape[IDX_TEMPORAL]):
            x_t = x_s[:,:,k_t]
            y_t = self.gcn(x_t, data.edge_index, data.edge_attr)
            y_t = F.relu(y_t)
            total.append(y_t)

        nodes_times = torch.stack(total, dim=IDX_TEMPORAL)

        return nodes_times

class STGCN(torch.nn.Module):
    '''The main SpatioTemporal Graph Convolutional Net class, based on temporal and spatial convolutions along
    with two layer normalizations and a fully connected layer at the end.

    Args:
        n_temporal_dim (int): Number of temporal units per spatial units as input
        n_spatial_dim (int): Number of spatial units
        n_input_channels (int): Number of data channels of input
        co_temporal (int, optional): Number of data output channels following a temporal convolution. Default 64.
        co_spatial (int, optional): Number of data output channels following a spatial convolution. Default 16.
        time_conv_length (int, optional): Length of temporal convolution window. Default 3.

    '''
    def __init__(self, n_temporal_dim, n_spatial_dim, n_input_channels,
                 co_temporal=64, co_spatial=16, time_conv_length=3):

        super(STGCN, self).__init__()

        self.n_temporal = n_temporal_dim
        self.n_spatial = n_spatial_dim
        self.n_input_channels = n_input_channels
        self.co_temporal = co_temporal
        self.co_spatial = co_spatial

        self.permute_norm = (IDX_CHANNEL, IDX_SPATIAL, IDX_TEMPORAL)

        assert n_temporal_dim - 4 * time_conv_length + 4 == 1

        self.model_t_1a = Time1dConvGLU(n_spatial_dim, n_temporal_dim,
                                        channel_inputs=n_input_channels,
                                        channel_outputs=co_temporal,
                                        time_convolution_length=time_conv_length)
        self.model_s_1 = SpatialGraphConv(n_spatial_dim, n_temporal_dim - time_conv_length + 1,
                                          channel_inputs=co_temporal,
                                          channel_outputs=co_spatial)
        self.model_t_1b = Time1dConvGLU(n_spatial_dim, n_temporal_dim - time_conv_length + 1,
                                        channel_inputs=co_spatial,
                                        channel_outputs=co_temporal,
                                        time_convolution_length=time_conv_length)
        self.layer_norm_1 = torch.nn.LayerNorm([n_spatial_dim, n_temporal_dim - 2 * time_conv_length + 2])

        self.model_t_2a = Time1dConvGLU(n_spatial_dim, n_temporal_dim - 2 * time_conv_length + 2,
                                        channel_inputs=co_temporal,
                                        channel_outputs=co_temporal,
                                        time_convolution_length=time_conv_length)
        self.model_s_2 = SpatialGraphConv(n_spatial_dim, n_temporal_dim - 3 * time_conv_length + 3,
                                          channel_inputs=co_temporal,
                                          channel_outputs=co_spatial)
        self.model_t_2b = Time1dConvGLU(n_spatial_dim, n_temporal_dim - 3 * time_conv_length + 3,
                                        channel_inputs=co_spatial,
                                        channel_outputs=co_temporal,
                                        time_convolution_length=time_conv_length)
        self.layer_norm_2 = torch.nn.LayerNorm([n_spatial_dim, n_temporal_dim - 4 * time_conv_length + 4])

        self.model_output = torch.nn.Sequential(torch.nn.Linear(in_features=n_spatial_dim * co_temporal,
                                                                out_features=n_spatial_dim * n_input_channels),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(in_features=n_spatial_dim * n_input_channels,
                                                                out_features=n_spatial_dim * n_input_channels))

    def forward(self, data_graph):

        data_step_0 = data_graph
        edge_index = data_graph.edge_index
        edge_attr = data_graph.edge_attr
        self.n_graphs_in_batch = self._infer_batch_size(data_step_0.x)

        data_step_0_x = data_step_0.x
        data_step_1_x = self.model_t_1a(data_step_0_x)
        data_step_1 = Data(data_step_1_x, edge_index, edge_attr)
        data_step_2_x = self.model_s_1(data_step_1)
        data_step_3_x = self.model_t_1b(data_step_2_x)
        data_step_3_x = self._compute_layer_norm(data_step_3_x, self.layer_norm_1)

        data_step_4_x = self.model_t_2a(data_step_3_x)
        data_step_4 = Data(data_step_4_x, edge_index, edge_attr)
        data_step_5_x = self.model_s_2(data_step_4)
        data_step_6_x = self.model_t_2b(data_step_5_x)
        data_step_6_x = self._compute_layer_norm(data_step_6_x, self.layer_norm_2)

        data_out_reshape = data_step_6_x.reshape(self.n_graphs_in_batch, self.n_spatial * self.co_temporal)
        data_output_x = self.model_output(data_out_reshape)
        data_output_x = data_output_x.reshape(self.n_graphs_in_batch * self.n_spatial, self.n_input_channels)

        return Data(y=data_output_x, edge_index=edge_index, edge_attr=edge_attr)

    def _infer_batch_size(self, data_x):
        '''Infer if data was produced in batches by Pytorch geometric, and if so, how many'''

        if data_x.shape[IDX_SPATIAL] % self.n_spatial == 0:
            n_batches = data_x.shape[IDX_SPATIAL] // self.n_spatial
        else:
            raise RuntimeError('A batch of graphs obtained with spatial dimension not an integer multiplier of initialized size {}'.format(self.n_spatial))

        return n_batches

    def _compute_layer_norm(self, data_x, norm_func):
        '''Compute the layer normalization over the channels for each spatio-temporal coordinate.

        The computation requires a temporary permutation of axes since the convention of
        `torch.nn.LayerNorm` is that the average and standard deviation are computed over the axes
        up to the normalized shape of the initialization. Each spatio-temporal coordinate should
        have its own normalization, hence the spatial and temporal axes have to be put last

        '''
        ret = []
        for k_batch in range(self.n_graphs_in_batch):
            data_x_pergraph = data_x.narrow(IDX_SPATIAL, k_batch * self.n_spatial, self.n_spatial)
            data_x_pergraph_norm = norm_func(data_x_pergraph.permute(self.permute_norm)).permute(self.permute_norm)
            ret.append(data_x_pergraph_norm)

        return torch.cat(ret)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

    stgcn = STGCN(9, 5, 2)
    print (count_parameters(stgcn.model_output))
    out = stgcn.forward(data_step_0)
    print (out)

if __name__ == '__main__':
    main()