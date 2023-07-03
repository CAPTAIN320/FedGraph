from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, BitcoinOTCDataset, RedditDataset
from data import *

from sybil_utils import data_split

import torch

data = CoraGraphDataset()
g = data[0]
# print(g)

split_method = "random_choice"
split = 100
graphs = data_split(g, split_method, split)

chosen_graph = graphs[0]
# Print graph information
print("num of nodes: ",chosen_graph.num_nodes())
print("num of edges: ",chosen_graph.num_edges())
print("is homogenous: ",chosen_graph.is_homogeneous)

def modify_g_edges(graph):
    # Clone the graph
    sybil_graph = graph.clone()

    for edge_index in range(graph.num_edges()):
        edge_id = graph.edata['_ID'][edge_index]
        sybil_id = edge_id + 2
        sybil_graph.edata['_ID'][edge_index] = sybil_id

    return sybil_graph

def sybil_add_nodes(graph, amount=0):
    graph.add_nodes(amount)
    return graph

def sybil_add_edges(graph, src_node_id, dest_node_id, no_of_new_edges=0, target_edge_feat='', new_edge_value=0): # CPT-NOTE: Node is added if src & dest nodes do NOT exist
    for i in range(no_of_new_edges):
        graph.add_edges(src_node_id, dest_node_id, {target_edge_feat: torch.tensor([new_edge_value])})
    return graph

# sybil_graph = add_g_edges(chosen_graph)
# print("num of sybil nodes: ",sybil_graph.num_nodes())
# print("num of sybil edges: ",sybil_graph.num_edges())

# print(sybil_graph.edata)

# sybil_graph = sybil_add_nodes(sybil_graph, 10)
# print("new num of sybil nodes: ",sybil_graph.num_nodes())
# print("new num of sybil edges: ",sybil_graph.num_edges())

sybil_graph = chosen_graph

# TODO: Understand more
sybil_graph = sybil_add_edges(sybil_graph, sybil_graph.num_nodes() - 1, sybil_graph.num_nodes() - 2)

print('List of Edge Indexes: ', chosen_graph.edata['_ID'].tolist())

no_of_new_edges = 5
source_node = sybil_graph.num_nodes() - 1
destination_node = sybil_graph.num_nodes() - 2
target_edge_feature = '_ID'
new_edge_value = 9

# sybil_graph = sybil_add_edges(sybil_graph, source_node, destination_node, no_of_new_edges)
sybil_graph = sybil_add_edges(sybil_graph, torch.tensor(5), torch.tensor([5,1]), no_of_new_edges, target_edge_feature, new_edge_value)
print("new num of sybil nodes: ",sybil_graph.num_nodes())
print("new num of sybil edges: ",sybil_graph.num_edges())
print('List of Sybil Edge Indexes: ', sybil_graph.edata['_ID'].tolist())
