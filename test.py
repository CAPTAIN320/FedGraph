from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, BitcoinOTCDataset, RedditDataset
from data import *

import torch

data = CoraGraphDataset()
g = data[0]
# print(g)

def data_split(g, args, split):
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
split = 2
graphs = data_split(g, split_method, split)
# print(graphs[99])

chosen_graph = graphs[1]
print("num of nodes: ",chosen_graph.num_nodes())
print("num of edges: ",chosen_graph.num_edges())
print("is homogenous: ",chosen_graph.is_homogeneous)
print("num of source nodes: ",chosen_graph.num_src_nodes())
print("num of destination nodes: ",chosen_graph.num_dst_nodes())

node_index = 0

feat = chosen_graph.ndata['feat'][node_index]
print("feat: ",feat)
sybil_feat = feat * 2
print("sybil_feat: ",sybil_feat)

label = chosen_graph.ndata['label'][node_index]
print("label: ", label)
sybil_label = label * 2
print("sybil_label: ", sybil_label)

val_mask = chosen_graph.ndata['val_mask'][node_index]
print("val_mask: ", val_mask)
sybil_val_mask = torch.tensor(True)
print("sybil_val_mask: ", sybil_val_mask)

test_mask = chosen_graph.ndata['test_mask'][node_index]
print("test_mask: ",test_mask)
sybil_test_mask = torch.tensor(True)
print("sybil_test_mask: ",sybil_test_mask)

train_mask = chosen_graph.ndata['train_mask'][node_index]
print("train_mask: ",train_mask)
sybil_train_mask = torch.tensor(True)
print("sybil_train_mask: ",sybil_train_mask)

id = chosen_graph.ndata['_ID'][node_index]
print("_ID: ",id)
sybil_id = id + 2
print("sybil_id: ",sybil_id)

edge_index = 0
edge_id = chosen_graph.edata['_ID'][edge_index]
print("edge id: ",edge_id)

chosen_graph.add_nodes(100) # add 100 nodes
print(chosen_graph.num_nodes())
print(chosen_graph.add_edges(0,1000)) # add edge from node 0 to node 1
print(chosen_graph.num_edges())
print()

# Modify node data
for node_index in range(chosen_graph.num_nodes()):
    feat = chosen_graph.ndata['feat'][node_index]
    print("feat: ",feat)
    sybil_feat = feat * 2
    print("sybil_feat: ",sybil_feat)

    label = chosen_graph.ndata['label'][node_index]
    print("label: ", label)
    sybil_label = label * 2
    print("sybil_label: ", sybil_label)

    val_mask = chosen_graph.ndata['val_mask'][node_index]
    print("val_mask: ", val_mask)
    sybil_val_mask = torch.tensor(True)
    print("sybil_val_mask: ", sybil_val_mask)

    test_mask = chosen_graph.ndata['test_mask'][node_index]
    print("test_mask: ", test_mask)
    sybil_test_mask = torch.tensor(True)
    print("sybil_test_mask: ", sybil_test_mask)

    train_mask = chosen_graph.ndata['train_mask'][node_index]
    print("train_mask: ",train_mask)
    sybil_train_mask = torch.tensor(True)
    print("sybil_train_mask: ",sybil_train_mask)

    id = chosen_graph.ndata['_ID'][node_index]
    print("_ID: ",id)
    sybil_id = id + 2
    print("sybil_id: ",sybil_id)
