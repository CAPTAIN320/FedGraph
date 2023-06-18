import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import dgl.function as f

# Multi-layer perceptron (MLP) for graph classification
class MLP(nn.Module):
    def __init__(self, in_feats, n_hidden, num_classes, n_layers, dropout):
        super(MLP, self).__init__()
        self.activation = F.relu
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, none, h):
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = self.activation(layer(h))
        return h

# Dot product predictor for link prediction
class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(f.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

# Graph convolutional network (GCN) for graph classification
class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 num_classes,
                 n_layers,
                 dropout):
        super(GCN, self).__init__()
        # self.g = g
        self.layers = nn.ModuleList()
        self.activation = F.relu
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=self.activation, allow_zero_in_degree=True))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=self.activation, allow_zero_in_degree=True))
        # output layer
        self.layers.append(GraphConv(n_hidden, num_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
