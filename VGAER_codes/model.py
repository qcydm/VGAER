import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from layer import GraphConvolution

class GAER(nn.Module):
    def __init__(self,  input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GAER, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1*2, dropout)
        self.gc2 = GraphConvolution(hidden_dim1*2, hidden_dim1, dropout)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout)
        self.gc4 = GraphConvolution(hidden_dim2, hidden_dim2, dropout)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.dropout = dropout

    def enconde(self, x, a_hat):

        hidden1 = F.tanh(self.gc1(input=x, adj=a_hat))
        hidden2 = F.tanh(self.gc2(input=hidden1, adj=a_hat))
        hidden3 = F.tanh(self.gc3(input=hidden2, adj=a_hat))
        hidden3 = self.gc4(input=hidden3, adj = a_hat)


        return hidden3

    def forward(self, A_hat,x):
        z = self.enconde(x=x, a_hat=A_hat)
        z = z.cuda()
        hid_emb = z
        return hid_emb, self.dc(z)

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        B_hat = z @ z.t()
        B_hat = F.sigmoid(B_hat)
        return B_hat


#定义VGAEModel
class VGAERModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim,device):#初始化VGAE
        super(VGAERModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        layers = [GraphConvolution(self.in_dim, self.hidden1_dim, act=F.tanh),
                  GraphConvolution(self.hidden1_dim, self.hidden2_dim, act=lambda x: x),
                  GraphConvolution(self.hidden1_dim, self.hidden2_dim, act=lambda x: x)]
        self.layers = nn.ModuleList(layers)
        self.device = device

        #三层GraphConv，原文中生成均值和方差的W0是共享的，W1是不同的，因此一共要三层
        #https://docs.dgl.ai/en/0.6.x/_modules/dgl/nn/pytorch/conv/graphconv.html
        #GraphConv用于实现GCN的卷积
        # layers = [self.gc1,self.gc2,self.gc2]#第二层求方差
        self.layers = nn.ModuleList(layers)

    def encoder(self, a_hat, features):

        h = self.layers[0](a_hat, features)
        self.mean = self.layers[1](a_hat, h)
        self.log_std = self.layers[2](a_hat, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(self.device)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))#解码器点乘还原邻接矩阵A'
        return adj_rec

    def forward(self,a_hat, features ):#前向传播
        z = self.encoder(a_hat,features )#编码器得到隐变量
        adj_rec = self.decoder(z)#解码器还原邻接矩阵
        return adj_rec,z
