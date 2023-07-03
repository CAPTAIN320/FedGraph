from data import *

# Split the graph
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
