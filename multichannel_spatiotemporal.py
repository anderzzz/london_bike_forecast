import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.data import Data

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
        print (x.shape)
        for k_node, tensor_node in enumerate(x.split(1, dim=IDX_SPATIAL)):
            print (tensor_node.shape)
            pq_out = self.one_d_conv_1(tensor_node)
            y_conv_and_gated_out = F.glu(pq_out, dim=1)
            total.append(y_conv_and_gated_out)

        nodes_times_nonnorm = torch.cat(total, dim=IDX_SPATIAL)
        print (nodes_times_nonnorm)
        print (nodes_times_nonnorm.shape)

        print (self.permute_norm)
        print (nodes_times_nonnorm.permute(self.permute_norm).shape)
        nodes_times_norm = self.layer_norm(nodes_times_nonnorm.permute(self.permute_norm)).permute(self.permute_norm)
        print (nodes_times_norm)
        print (nodes_times_norm.shape)

        # Layer normalization. Each output channel is normalized with distinct parameters, mean and std obtained
        # over the plurality of nodes and time steps the channel processes. Due to how dimensions of tensor
        # may be ordered, a permutation of dimensions may be required.
        #nodes_times_norm = F.layer_norm(nodes_times_nonnorm.permute(self.dim_permute_norm),
        #                                [nodes_times_nonnorm.shape[self.dim_node],
        #                                 nodes_times_nonnorm.shape[self.dim_timestep]]).permute(self.dim_permute_norm)

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
        print (nodes_times_nonnorm.shape)

        nodes_times_norm = self.layer_norm(nodes_times_nonnorm.permute(self.permute_norm)).permute(self.permute_norm)
        print (nodes_times_norm)
        print (nodes_times_norm.shape)

        # Layer normalization. Each output channel is normalized with distinct parameters, mean and std obtained
        # over the plurality of nodes and time steps the channel processes. Due to how dimensions of tensor
        # may be ordered, a permutation of dimensions may be required.
#        nodes_times_norm = F.layer_norm(nodes_times_nonnorm.permute(self.dim_permute_norm),
#                                        [nodes_times_nonnorm.shape[self.dim_node],
#                                         nodes_times_nonnorm.shape[self.dim_timestep]]).permute(self.dim_permute_norm)
#        print (nodes_times_norm.shape)

        return nodes_times_norm

def construct_data():

    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 2, 4],
                               [1, 0, 2, 1, 3, 2, 4, 2]], dtype=torch.long)
    edge_weight = None
    x = torch.tensor([[[1,1,4,1,6],[20,20,70,10,70]],
                      [[2,3,0,1,0],[50,50,60,10,10]],
                      [[3,2,2,8,9],[40,40,30,40,40]],
                      [[5,4,3,3,2],[50,40,40,50,80]],
                      [[9,0,9,2,8],[60,70,70,20,20]]], dtype=torch.float)
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

    model_t1 = Time1dConvGLU(n_spatial=5, n_temporal=5,
                             channel_inputs=2, channel_outputs=4,
                             time_convolution_length=3)
    model_s = SpatialGraphConv(n_spatial=5, n_temporal=3,
                               channel_inputs=4, channel_outputs=2)
    model_t2 = Time1dConvGLU(n_spatial=5, n_temporal=3,
                             channel_inputs=2, channel_outputs=2,
                             time_convolution_length=3)

    print ('AA1')
    print (data_step_0.x.shape)
    data_step_1_x = model_t1(data_step_0.x)
    data_step_1 = Data(x=data_step_1_x, edge_index=edge_index, edge_attr=edge_attr)

    print ('AA2')
    data_step_2_x = model_s(data_step_1)
    print ('AA3')
    data_step_3_x = model_t2(data_step_2_x)
    print ('AA4')
    print (data_step_3_x.shape)

if __name__ == '__main__':
    main()