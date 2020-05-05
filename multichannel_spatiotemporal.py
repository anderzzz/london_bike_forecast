import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.data import Data

class Time1dConvGLU(torch.nn.Module):
    '''Convolution along time dimension with gated linear unit as output activation

    Args:
        channel_inputs (int): Number of channels of input per node
        channel_outputs (int): Number of channels of output per node
        time_convolution_length (int): The kernel size of the time convolution window
        dim_node (int, Optional): The dimension for the nodes on the data. Default is 0
        dim_channels_input (int, Optional): The dimension for the input channels on the data. Default is 1
        dim_timestep (int, Optional): The dimension for the time steps on the data. Default is 2

    '''
    def __init__(self, channel_inputs, channel_outputs, time_convolution_length,
                 dim_node=0, dim_channels_input=1, dim_timestep=2):

        super(Time1dConvGLU, self).__init__()

        self.channel_inputs = channel_inputs
        self.channel_outputs = channel_outputs

        self.dim_node = dim_node
        self.dim_channels_input = dim_channels_input
        self.dim_timestep = dim_timestep

        self.one_d_conv_1 = torch.nn.Conv1d(in_channels=self.channel_inputs,
                                            out_channels=2 * self.channel_outputs,
                                            kernel_size=time_convolution_length,
                                            groups=1)

    def forward(self, x):
        '''Forward operation'''

        total = []
        for k_node, tensor_node in enumerate(x.split(1, dim=self.dim_node)):
            pq_out = self.one_d_conv_1(tensor_node)
            y_conv_and_gated_out = F.glu(pq_out, dim=1)
            total.append(y_conv_and_gated_out)

        ret_tensor = torch.cat(total, dim=self.dim_node)

        return ret_tensor

class SpatialGraphConv(torch.nn.Module):

    def __init__(self, n_nodes, channel_inputs, channel_outputs,
                 dim_node = 0, dim_channels_input = 1, dim_timestep = 2):

        super(SpatialGraphConv, self).__init__()

        self.n_nodes = n_nodes
        self.channel_inputs = channel_inputs
        self.channel_outputs = channel_outputs

        self.dim_node = dim_node
        self.dim_channels_input = dim_channels_input
        self.dim_timestep = dim_timestep

#        gcns = []
#        for i in range(self.channel_inputs):
#            for j in range(self.channel_inputs):
#                gcns.append(GCNConv(64, 16))
#        self.gcn_convs = torch.nn.ModuleList(gcns)
        self.gcn = GCNConv(self.channel_inputs, self.channel_outputs)

    def forward(self, data):

        assert self.channel_inputs == data.x.shape[self.dim_channels_input]

        x_s = data.x
        total = []
        for k_t in range(data.x.shape[self.dim_timestep]):
            x_t = x_s[:,:,k_t]
            y_t = self.gcn(x_t, data.edge_index, data.edge_attr)
            total.append(y_t)

        ret_tensor = torch.stack(total, dim=self.dim_timestep)

        return ret_tensor

def construct_data():

    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 2, 4],
                               [1, 0, 2, 1, 3, 2, 4, 2]], dtype=torch.long)
    edge_weight = None
    x = torch.tensor([[[1,1,4,1,6,7],[2,2,7,0,7,6]],
                      [[2,3,0,1,0,0],[5,5,6,0,0,1]],
                      [[3,2,2,8,9,1],[4,4,3,4,4,4]],
                      [[5,4,3,3,2,1],[5,4,4,5,8,8]],
                      [[9,0,9,2,8,3],[6,7,7,2,2,1]]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    return data

def main():

    data_step_0 = construct_data()
    edge_index = data_step_0.edge_index
    edge_attr = data_step_0.edge_attr

    model_t1 = Time1dConvGLU(2, 64, 3)
    model_s = SpatialGraphConv(5, 64, 16)
    model_t2 = Time1dConvGLU(16, 2, 3)

    data_step_1_x = model_t1(data_step_0.x)
    data_step_1 = Data(x=data_step_1_x, edge_index=edge_index, edge_attr=edge_attr)

    data_step_2_x = model_s(data_step_1)
    data_step_3_x = model_t2(data_step_2_x)

    print (data_step_3_x.shape)

if __name__ == '__main__':
    main()