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

def add_g_edges(graph):
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

def sybil_add_edges(graph, src_node_id, dest_node_id): # CPT-NOTE: Node is added if src & dest nodes do NOT exist
    graph.add_edges(src_node_id, dest_node_id)
    return graph

sybil_graph = add_g_edges(chosen_graph)
print("num of sybil nodes: ",sybil_graph.num_nodes())
print("num of sybil edges: ",sybil_graph.num_edges())

print(sybil_graph.edata)

# sybil_graph = sybil_add_nodes(sybil_graph, 10)
# print("new num of sybil nodes: ",sybil_graph.num_nodes())
# print("new num of sybil edges: ",sybil_graph.num_edges())

# TODO: Understand more
sybil_graph = sybil_add_edges(sybil_graph, 0, 1)
print("new num of sybil nodes: ",sybil_graph.num_nodes())
print("new num of sybil edges: ",sybil_graph.num_edges())

print(sybil_graph.edata)
