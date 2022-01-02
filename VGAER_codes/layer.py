# import torch
# import torch.nn.functional as F
# from torch.nn.modules.module import Module
# from torch.nn.parameter import Parameter
#
#
# class GraphConvolution(Module):
#     def __init__(self, in_features, out_features, dropout=0., act=F.relu):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.act = act
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#
#     def forward(self, x, A):
#         x = F.dropout(x, self.dropout, self.training)
#         mid = x @ self.weight
#         output = A @ mid
#         # output = self.act(output) #act=F.relu activate
#         return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'


import math
import torch.nn.functional as F
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, act=F.tanh):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.act = act
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        self.weight.cuda()
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        # 如果使用vgaer，需要使用激活函数
        # output = self.act(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
