import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from torch_geometric.data import Data

class Net(torch.nn.Module):

    def __init__(self):
        '''Bla bla

        '''
        super(Net, self).__init__()

    def forward(self):

        pass

class Time1dConvGLU(torch.nn.Module):
    '''Convolution along time dimension with gated linear unit as output activation

    The module is comprised of two parts: depthwise one-dimensional convolution and gated linear activation

    The depthwise one-dimensional convolution performs for each node in the graph two parallel convolutions of the
    time dimension of a set convolution window size, `time_convolution_length`. The depthwise property means
    the output for any given node depends only on the values for that node along the time dimension. This is a
    temporal operation exclusively.

    The gated linear activation, for a given node, takes the values of the two parallel convolutions and computes
    the gated linear activation.

    The output of a forward operation of the module is therefore of a dimension: (number of batches) X
    (number of nodes in graph) X (number of time steps - convolution window size).

    Args:
        channel_inputs (int): Number of input channels, which should be equal to the number of nodes of graph
        time_convolution_length (int, optional): Convolution window size. Defaults to 3

    '''
    def __init__(self, channel_inputs, time_convolution_length=3):

        super(Time1dConvGLU, self).__init__()

        self.channel_inputs = channel_inputs
        self.channel_outputs = 2 * channel_inputs

        self.one_d_conv_1 = torch.nn.Conv1d(in_channels=self.channel_inputs,
                                            out_channels=self.channel_outputs,
                                            kernel_size=time_convolution_length,
                                            groups=channel_inputs)

    def forward(self, x):
        '''Forward operation'''

        pq_out = self.one_d_conv_1(x)

        # Output order is:
        # (node 1, first output channel), (node 1, second output channel),
        # (node 2, first output channel), (node 2, second output channel) ...
        # However, the GLU splits the input tensor in half. In order to have second output channel for node X
        # to gate the tensor value for the first output channel for node X, the tensor has to be reordered
        # accordingly for the second dimension
        inds = list(range(0, self.channel_outputs, 2)) + list(range(1, self.channel_outputs + 1, 2))
        pq_out_reorder = pq_out[:, torch.LongTensor(inds), :]

        y_conv_and_gated_out = F.glu(pq_out_reorder, dim=1)

        return y_conv_and_gated_out

class SpatialGraphConv(torch.nn.Module):

    def __init__(self, n_nodes, n_data_per_node=1, cheby_kernel_size=2):

        super(SpatialGraphConv, self).__init__()

        self.n_nodes = n_nodes
        self.n_data_per_node = n_data_per_node
        self.cheby_kernel_size = cheby_kernel_size

        self.cheby_graph_conv = {}
        for i in range(self.n_data_per_node):
            for j in range(self.n_data_per_node):
                self.cheby_graph_conv[(i,j)] = ChebConv(in_channels=self.n_nodes,
                                                        out_channels=self.n_nodes,
                                                        K=self.cheby_kernel_size,
                                                        node_dim=1)

    def forward(self, data):
        '''Forward bla bla

        Args:
            data (Tensor): graph, with node data in data.x. Dimensions are (batch, node, node_data_type, time_step)

        '''
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        print (data)
        print (x)
        print (x.shape)
        x_ = torch.unsqueeze(x, 0)
        shape_per_input_channel = x_[:,:,0,:].shape
        print (x_)
        print (x_.shape)
        print (edge_index)
        print (edge_weight)

        print (x_[:,:,0,:])
        print (x_[:,:,0,:].shape)
        print (torch.zeros(x_[:,:,0,:].shape))

        print ('AAA1')
        for i in range(self.n_data_per_node):
            y_i = torch.zeros(shape_per_input_channel)
            for j in range(self.n_data_per_node):
                x_j = x_[:,:,j,:]
                print (x_j)
                ttt= self.cheby_graph_conv[(i,j)].node_dim
                print (ttt)
                print (x_j.size(ttt))
                y_ij = self.cheby_graph_conv[(i, j)](x_j, edge_index, edge_weight)
                y_i += y_ij
            y_i = F.relu(y_i)
            raise RuntimeError

        raise RuntimeError

def construct_data():

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    #x = torch.tensor([[7,5,6,1,8,0], [5,1,1,1,8,7], [3,0,2,8,7,2]], dtype=torch.float)
    x = torch.tensor([[[8,1,4,1],[2,2,7,0]], [[2,3,0,1],[5,5,6,0]], [[10,2,2,8],[4,4,3,4]]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    return data

def main():

    data = construct_data()
    #print (data)
    #print (data.x)
    #print (data.x.shape)
    #x_ = torch.unsqueeze(data.x, 0)
    #model_t = Time1dConvGLU(3, 3)
    #model_t(x_)

    model_s = SpatialGraphConv(3, n_data_per_node=2)
    print (model_s)
    print (dir(model_s))
    model_s(data)

#    from torch.utils.tensorboard import SummaryWriter
#
#    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
#    writer.add_graph(model, x_)
#    writer.close()

if __name__ == '__main__':
    main()
