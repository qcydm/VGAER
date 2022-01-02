from __future__ import division
from __future__ import print_function

import argparse
import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import pandas as pd
# import scanpy
import csv

from dgl.data import CoraGraphDataset, CiteseerGraphDataset , PubmedGraphDataset


from model import GAER
# from optimizer import loss_function
# import torch.nn.modules.loss
import torch.nn.functional as F
# from algorithm

from utils import load_data, loadppi

from cluster import community
from NMI import load_label, NMI, label_change
from Qvalue import Q
from tsne import get_data,tsne_show
from Qwepoch import Q_with_epoch

figcount = 0
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--cluster', type=str, default=7, help='Number of community')


args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def gaer_for(args):


    # data:Cora
    # print("Using {} dataset".format(args.dataset_str))
    # features, adj = load_data(args.dataset_str)
    # dataset = CoraGraphDataset(reverse_edge=False)
    # graph = dataset[0]
    # A = graph.adjacency_matrix().to_dense()
    # # A = adj
    # A_orig_ten = A
    # A_orig = A.detach().numpy()
    # features = graph.ndata['feat']
    # features = features.to(device)
    # label_orig = graph.ndata['label'].detach().numpy()


    # data:Karate
    # G = nx.karate_club_graph()
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # feature_dim = 34  # 每个结点的特征维度
    # features = torch.eye(34, feature_dim)
    #
    # A_orig = A.detach().numpy()
    # A_orig_ten = A

    # data:Dolphin social network
    # columns = ['Source', 'Target']
    # data = pd.read_csv('data/dolphins.csv', names=columns, header=None)
    # G = nx.Graph()
    # data_len = len(data)
    # for i in range(data_len):
    #     G.add_edge(data.iloc[i]['Source'], data.iloc[i]['Target'])
    #
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # feature_dim = 62  # 每个结点的特征维度
    # features = torch.eye(feature_dim, feature_dim)
    #
    # A_orig = A.detach().numpy()

    # data:footballs
    # G = nx.read_gml('data/football.gml')
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # k=12
    # sc=algorithm.SpectralC
    # feature_dim = A.shape[0]
    # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A



    # # data:ppi
    #
    # A = loadppi()
    # features = np.load('data/ppi/ppi-feats.npy')
    # features = torch.Tensor(features)
    # A_orig = A.detach().numpy()


    # data:power
    # G = nx.read_gml('data/power/power.gml', label='id')
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # feature_dim = A.shape[0]
    # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A


    # # data:lesmis
    # G = nx.read_gml('data/lesmis/lesmis.gml', label='id')
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # # feature_dim = A.shape[0]
    # # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A
    #
    # data:deezer_europe_edges
    # columns = ['Source', 'Target']
    # data = pd.read_csv('data/deezer_europe/deezer_europe_edges.csv', names=columns, header=None)
    # G = nx.Graph()
    # data_len = len(data)
    # for i in range(data_len):
    #     G.add_edge(data.iloc[i]['Source'], data.iloc[i]['Target'])
    #
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # A_orig = A.detach().numpy()

    #  # data:celegansneural
    # G = nx.read_gml('data/celegansneural/celegansneural.gml')
    # G = nx.read_graphml('data/celegansneural/celegansneural.gml')
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # feature_dim = A.shape[0]
    # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A

    # data:adj_noun
    # adata = scanpy.read('data/adjnoun/adjnoun.mtx')
    # G = adata.X
    # A = torch.Tensor(G.A)
    # feature_dim = A.shape[0]
    # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A


    # data:lastfm_asia
    # columns = ['Source', 'Target']
    # data = pd.read_csv('data/lastfm_asia/lastfm_asia_edges.csv', names=columns, header=None)
    # G = nx.Graph()
    # data_len = len(data)
    # for i in range(data_len):
    #     G.add_edge(data.iloc[i]['Source'], data.iloc[i]['Target'])
    #
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # # feature_dim = A.shape[0]
    # # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A

    # G = nx.read_gml('data/power/power.gml', label='id')
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # A_orig = A.detach().numpy()
    # A_orig_ten = A
    # A_orig_ten = A_orig_ten.to(device)


    # # data:Roget
    # adata = scanpy.read('data/Roget/Roget.mtx')
    # G = adata.X
    # A = torch.Tensor(G.A)
    # A_orig = A.detach().numpy()
    # feature_dim = A.shape[0]
    # features = torch.eye(feature_dim, feature_dim)
    # A_orig_ten = A
    #
    #
    # data: netscience
    # adata = scanpy.read('data/netscience/netscience.mtx')
    # G = adata.X
    # A = torch.Tensor(G.A)
    # # feature_dim = A.shape[0]
    # # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A

    # # data: econ-poli
    # adata = scanpy.read('data/ca-Erdos992/Erdos992.mtx')
    # G = adata.X
    # A = torch.Tensor(G.A)
    # feature_dim = A.shape[0]
    # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A

    # data: rt
    # G = nx.read_edgelist('data/rt_dash/rt_dash.edges')
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # feature_dim = A.shape[0]
    # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A

    # ca
    # columns = ['Source', 'Target']
    # data = pd.read_csv('data/ca-GrQc/ca-GrQc123.csv', names=columns, header=None)
    # G = nx.Graph()
    # data_len = len(data)
    # for i in range(data_len):
    #     G.add_edge(data.iloc[i]['Source'], data.iloc[i]['Target'])
    #
    # A = torch.Tensor(nx.adjacency_matrix(G).todense())
    # feature_dim = A.shape[0]
    # features = torch.eye(feature_dim, feature_dim)
    # A_orig = A.detach().numpy()
    # A_orig_ten = A


    #
    #数据处理
    # 处理A矩阵，得到B

    if args.dataset == 'cora':
        dataset = CoraGraphDataset(reverse_edge=False)
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset(reverse_edge=False)
    elif args.dataset == 'pumbed':
        dataset = PubmedGraphDataset(reverse_edge=False)
    else:
        raise NotImplementedError
    graph = dataset[0]
    A = graph.adjacency_matrix().to_dense()
    A_orig = A.detach().numpy()
    # A_orig_ten = A
    label_orig = graph.ndata['label'].detach().numpy()

    K = 1 / (A.sum().item()) * (A.sum(dim=1).reshape(A.shape[0], 1) @ A.sum(dim=1).reshape(1, A.shape[0]))
    B = A - K
    B = B.to(device)
    # features = B
    # Extract node features
    features = graph.ndata['feat']

    features = features.to(device)
    feats = torch.cat((features, B), 1)
    in_dim = feats.shape[-1]

    A = A + torch.eye(A.shape[0])
    D = torch.diag(torch.pow(A.sum(dim=1), -0.5))  # D = D^-1/2
    A_hat = D @ A @ D
    A_hat = A_hat.to(device)
    n_nodes, feat_dim = features.shape

    model = GAER(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    # nmi=[]






    # 整张图真实的空手道俱乐部人员的分类标签
    # class0idx = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21]
    # label_orig = [0 if i in class0idx else 1 for i in range(34)]

    # # # 整张图真实的海豚的分类标签
    # class0idx = [1, 5, 6, 7, 9, 13, 17, 19, 22, 25, 26, 27, 31, 32, 41, 48, 54, 56, 57, 60]
    # label_orig = [0 if i in class0idx else 1 for i in range(feature_dim)]



    for epoch in range(args.epochs):

        model.train()
        opt.zero_grad()
        recovered = model(A_hat=A_hat, x=features)
        loss = F.mse_loss(input=recovered[1], target=B)

        # loss = F.binary_cross_entropy_with_logits(input=F.sigmoid(recovered[1]), target=B)
        cur_loss = loss.item()
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss))
        loss.backward()
        opt.step()
        # model.eval()
        if epoch == args.epochs - 1:
            hidemb = recovered[0].cpu()
            hidemb = hidemb.cpu()
            commu_pred = community(hidemb, args.cluster)
            # Q_NUMBER = []
            # for i in range(5, args.cluster):
            #     commu_pred = community(hidemb, i)
            #     Q_NUMBER.append(Q(A_orig, np.eye(args.cluster)[commu_pred]))
            # print(Q_NUMBER)
            NMI(commu_pred, label_orig)
        # hidemb = recovered[0].cpu()
        # hidemb = hidemb.detach().numpy()
        # commu_pred = community(hidemb, args.cluster)
        # #
        # # # nminew=NMI(commu_pred, label_orig)
        # Q(A_orig, np.eye(args.cluster)[np.array(commu_pred)])
        # Q_value.append(Qnew)
        # nmi.append(nminew)

        # if epoch == args.epochs - 1:
        #     # Q_with_epoch(args.epochs, Q_value)
        #     # tsne_show(hidemb, commu_pred,figcount)
        #     commu_pred = community(hidemb, args.cluster)
        #     label_orig = load_label(args.dataset_str)[0]
        #     commu_pred2 = label_change(commu_pred, label_orig)
            # NMI(commu_pred2, label_orig)
            # for qi in range(5, 10):
            #     hidemb = recovered[0].detach().numpy()
            #     commu_pred = community(hidemb, qi)
            #     Q(A_orig, np.eye(qi)[np.array(commu_pred)])
        #
        # with open('karate_NMI500gaer.csv','a',encoding='utf-8', newline='') as csvfile:
        #     fieldnames = ['epoch', 'Qvalue']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writerow({'epoch': epoch, 'Qvalue': Qnew})
        # # with open('hidemb_gaer500(2).csv','a',encoding='utf-8', newline='') as csvfile:
        #     fieldnames = ['epoch', 'hidemb']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writerow({'epoch': epoch, 'hidemb': hidemb})


        # print('每个人的预测类别：', torch.Tensor(commu_pred2))
        # print('准确率：', float((torch.Tensor(commu_pred2) == adj_label).sum()) / A.shape[0])

        # tsne_show(hidemb, commu_pred2)

    print("Optimization Finished!")


if __name__ == '__main__':
        gaer_for(args)







