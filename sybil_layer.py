from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, BitcoinOTCDataset, RedditDataset
from data import *

import torch

data = CoraGraphDataset()
g = data[0]
# print(g)

# Split the graph
def data_split_2(g, args, split):
    if args == 'random_choice':
        # random_choice
        assign = random_choice(split, g.number_of_nodes()).tolist()
        index = [[] for i in range(split)]
        [index[ind].append(i) for i, ind in enumerate(assign)]

    graphs = [node_subgraph(g, index[i]) for i in range(split)]

    for i in range(len(graphs)):
        # graphs[i].int().to(args.device)
        # add self loop
        graphs[i] = remove_self_loop(graphs[i])
        graphs[i] = add_self_loop(graphs[i])

    return graphs

split_method = "random_choice"
split = 1
graphs = data_split_2(g, split_method, split)

chosen_graph = graphs[0]

def modify_g_node_values(graph):
    # Print graph information
    # print("num of nodes: ",graph.num_nodes())
    # print("num of edges: ",graph.num_edges())
    # print("is homogenous: ",graph.is_homogeneous)

    # Clone the graph
    sybil_graph = graph.clone()

    # Modify node data
    for node_index in range(graph.num_nodes()):
        # Get and modify feat
        feat = graph.ndata['feat'][node_index]
        sybil_feat = feat * 2
        sybil_graph.ndata['feat'][node_index] = sybil_feat

        # Get and modify label
        label = graph.ndata['label'][node_index]
        sybil_label = label * 2
        sybil_graph.ndata['label'][node_index] = sybil_label

        # Get and modify val_mask
        val_mask = graph.ndata['val_mask'][node_index]
        sybil_val_mask = torch.tensor(True)
        sybil_graph.ndata['val_mask'][node_index] = sybil_val_mask

        # Get and modify test_mask
        test_mask = graph.ndata['test_mask'][node_index]
        sybil_test_mask = torch.tensor(True)
        sybil_graph.ndata['test_mask'][node_index] = sybil_test_mask

        # Get and modify train_mask
        train_mask = graph.ndata['train_mask'][node_index]
        sybil_train_mask = torch.tensor(True)
        sybil_graph.ndata['train_mask'][node_index] = sybil_train_mask

        # Get and modify node ID
        id = graph.ndata['_ID'][node_index]
        sybil_id = id + 2
        sybil_graph.ndata['_ID'][node_index] = sybil_id

    return sybil_graph

def modify_g_edge_values(graph):
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


sybil_graph = modify_g_node_values(chosen_graph)
print("num of sybil nodes: ",sybil_graph.num_nodes())
print("num of sybil edges: ",sybil_graph.num_edges())

sybil_graph = modify_g_edge_values(chosen_graph)
print("num of sybil nodes: ",sybil_graph.num_nodes())
print("num of sybil edges: ",sybil_graph.num_edges())

sybil_graph = sybil_add_nodes(sybil_graph, 10)
print("new num of sybil nodes: ",sybil_graph.num_nodes())
print("new num of sybil edges: ",sybil_graph.num_edges())

# TODO: Understand more
# sybil_graph = sybil_add_edges(sybil_graph, sybil_graph[0], 1)
# print("new num of sybil nodes: ",sybil_graph.num_nodes())
# print("new num of sybil edges: ",sybil_graph.num_edges())
