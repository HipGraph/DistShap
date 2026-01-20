# Source: https://github.com/tamararruda/DnX/tree/main/FastDnX
# We modified the code to utilize sparse matrix instead of dense matrix.

import torch
import torch.nn as nn
import numpy as  np
from torch_geometric.utils import k_hop_subgraph

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from torch_geometric.nn import SGConv


class FastDnXSurrogateModel(torch.nn.Module):
    def __init__(self, in_features, num_classes, nhops=2):
        super().__init__()
        self.conv1 = SGConv(in_features, num_classes, K=nhops)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x
    




def A_k_hop(A, hop):
    edge_index, weight_edge = dense_to_sparse(A) 

    edge_index, edge_weight = gcn_norm(edge_index.long(), add_self_loops=True) 

    n_A = torch.zeros(A.shape) 

    for i, j in enumerate(edge_index.T): 
        n_A[j[0].item()][j[1].item()] = edge_weight[i]

    A_pot = n_A
    for i in range(hop-1):
        A_pot = torch.matmul(n_A, A_pot) 
    return A_pot

def sparse_A_k_hop(edge_index, num_nodes, hop):
    edge_index, edge_weight = gcn_norm(edge_index, add_self_loops=True)
    n_A = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    A_pot = n_A
    for i in range(hop-1):
        A_pot = torch.matmul(n_A, A_pot)
    return A_pot

class FastDnX():
    def __init__(self, model_to_explain, features, task, hop, edge_index):
        self.features = features
        self.hop = hop
        self.edge_index = edge_index
        self.labels = model_to_explain(features, edge_index)
        self.num_nodes = len(features)
        self.A_pot = []
        self.params = []
        self.model_to_explain = model_to_explain

        if task == "node":
            print('finding top nodes...')
        elif task == "edge":
            print('finding top nodes...')
        else:
            pass


    def prepare(self, ):
        # A = torch.zeros((self.num_nodes, self.num_nodes))
        # r, c = self.edge_index
        # A[r, c] = 1
        # self.A_pot = A_k_hop(A, self.hop)

        # use sparse matrix
        self.sparse_A_pot = sparse_A_k_hop(self.edge_index, self.num_nodes, self.hop)

        # validate if A_pot and sparse_A_pot are the same
        #print(torch.allclose(self.A_pot,self.sparse_A_pot.to_dense()))

        self.params = []
        for param in self.model_to_explain.parameters():
            self.params.append(param)

    def explain(self, node_to_be_explain):
        nodes_neigh = k_hop_subgraph(node_to_be_explain, self.hop, self.edge_index)[0]
        
        labels_node_expl = torch.zeros(self.labels[nodes_neigh].shape, device=nodes_neigh.device)
        # print(labels_node_expl + self.labels[node_to_be_explain], self.params[1].shape)
        labels_node_expl = labels_node_expl + self.labels[node_to_be_explain] - self.params[1]
        
        #X_pond = (self.features * self.A_pot[node_to_be_explain].view(-1,1))[nodes_neigh]
        #a = torch.matmul(X_pond, self.params[0].T)
        #expl = torch.diag(torch.matmul(a, labels_node_expl.T))
        #expl[torch.where(nodes_neigh==node_to_be_explain)] = expl.sum()

        # use sparse matrix
        X_pond_sparse = (self.features * self.sparse_A_pot[node_to_be_explain].to_dense().view(-1,1))[nodes_neigh]
        sparse_a = torch.matmul(X_pond_sparse, self.params[0].T)
        expl_sparse = torch.diag(torch.matmul(sparse_a, labels_node_expl.T))
        expl_sparse[torch.where(nodes_neigh==node_to_be_explain)] = expl_sparse.sum()

        # print(torch.allclose(expl, expl_sparse))
        # return nodes_neigh, expl.detach()
        return nodes_neigh, expl_sparse.detach()