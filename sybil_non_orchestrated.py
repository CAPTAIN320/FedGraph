from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, BitcoinOTCDataset, RedditDataset
from data import *

from sybil_utils import data_split

import torch
from random import randint

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

# def sybil_add_nodes(graph, amount=0):
#     graph.add_nodes(amount)
#     return graph

def add_sybil_edges(
                graph,
                src_node_id, # Should be less than no. of nodes
                dest_node_id, # Should be less than no. of nodes
                target_edge_feat, # Should be a valid edge feature name
                new_edge_value,
                no_of_new_edges=0,
            ): # CPT-NOTE: Node is added if src & dest nodes do NOT exist
    for i in range(no_of_new_edges):
        graph.add_edges(src_node_id, dest_node_id, {target_edge_feat: torch.tensor([new_edge_value])})
    return graph

# print('List of Edge Indexes: ', chosen_graph.edata['_ID'].tolist())

# sybil_graph = chosen_graph

# no_of_new_edges = 5
# source_node = sybil_graph.num_nodes() - 1
# destination_node = sybil_graph.num_nodes() - 2
# target_edge_feature = '_ID'
# new_edge_value = randint(0,10000)

# sybil_graph = add_edges(
#                             sybil_graph,
#                             source_node, # ShouldLess than the amount of nodes in the graph
#                             destination_node,
#                             target_edge_feature,
#                             new_edge_value,
#                             no_of_new_edges,
#                         )

# print("new num of sybil nodes: ",sybil_graph.num_nodes())
# print("new num of sybil edges: ",sybil_graph.num_edges())
# print('List of Sybil Edge Indexes: ', sybil_graph.edata['_ID'].tolist())
