import dgl
from models import *
from utils import *
from fl import *
from data import *
from args import *
from plot import *
import os
import random

from sybil_orchestrated import modify_g_node_values, modify_g_edge_values
from sybil_non_orchestrated import add_sybil_edges

import csv
import numpy as np

args = args()

def save_accuracy_csv(recorder, filename, args):
    test_acc = recorder['test_acc']['clients'][0]
    headers = ['Epoch', 'Accuracy', 'Sybil %']
    data = [(epoch+1, acc, args.num_sybils) for epoch, acc in enumerate(test_acc)]
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

# Initialize
os.makedirs('saves', exist_ok=True)
os.makedirs('results', exist_ok=True)

setup_seed(args.seed)
recorder = {'train_loss': {'clients': [[] for k in range(args.split)], 'server': []},
            'val_loss': {'clients': [[] for k in range(args.split)], 'server': []},
            'test_loss': {'clients': [[] for k in range(args.split)], 'server': []},
            'train_acc': {'clients': [[] for k in range(args.split)], 'server': []},
            'val_acc': {'clients': [[] for k in range(args.split)], 'server': []},
            'test_acc': {'clients': [[] for k in range(args.split)], 'server': []}}
# Data load & split
g = data_load(args)
e = g.edges(form='eid')
graphs = data_split(g, args)
# subfigs(graphs,args)


# FL initialize
server = Server(g, args)
clients = [Client(k, graphs[k], args) for k in range(args.num_clients)]

total_clients = len(clients)
num_sybil_clients = args.num_sybils
sybil_clients = random.sample(range(total_clients), num_sybil_clients)

attack_type = 'orchestrated'
num_fake_edges = 10

for index, client in enumerate(clients):
    if index in sybil_clients:
        print("Sybil Client ", index+1)
        print("Modifying Graph")
        if attack_type == 'orchestrated':
            client.g = modify_g_node_values(client.g) # modify node values
            client.g = modify_g_edge_values(client.g) # modify edge values
        else:
            no_of_new_edges = num_fake_edges
            source_node = client.g.num_nodes() - 1
            destination_node = client.g.num_nodes() - 2
            target_edge_feature = '_ID'
            # new_edge_value = random.randint(0,100)
            new_edge_value = 10

            client.g = add_sybil_edges(
                                        client.g,
                                        source_node, # Should be Less than the amount of nodes in the graph
                                        destination_node,
                                        target_edge_feature,
                                        new_edge_value,
                                        no_of_new_edges,
                                    )
            # print("new num of sybil nodes: ",client.g.num_nodes())
            # print("new num of sybil edges: ",client.g.num_edges())
            # print('List of Sybil Edge Indexes: ', client.g.edata['_ID'].tolist())
    else:
        print("Honest Client ", index+1)
    client.g.remove_edges(client.g.edges(form='eid'))
    # client.g = add_self_loop(client.g)
    add_edges(client.g)
    # print(client.g.num_edges())

# This client is used to evaluate the model on unseen data
test_client = Client(-1, graphs[-1], args)
test_client_acc = []

# Federated Learning
for _ in range(int(args.n_epochs)):
    for k in range(len(clients)):
        # Fork
        clients[k].fork(server)
        # Evaluate
        acc = evaluate(clients[k].model, clients[k], mask='test')
        recorder['test_acc']['clients'][k+1].append(acc)
        # Local_update
        clients[k].local_update()
    # Merge
    server.merge(clients)
    acc = evaluate(server.model, server)
    recorder['test_acc']['server'].append(acc)
    # Evaluate Test Client
    test_client_acc.append(evaluate(server.model, test_client))
    acc = evaluate(server.model, test_client)
    recorder['test_acc']['clients'][0].append(acc)

recorder['test_acc']['test_client'] = test_client_acc
# Attack 1
# save_accuracy_csv(recorder, f'./results/A1_{args.dataset}_{args.num_sybils}_sybils.csv', args)

# Attack 2
save_accuracy_csv(recorder, f'./results/A2_{args.dataset}_{args.num_sybils}_sybils_{num_fake_edges}_same-values.csv', args)
# CPT-NOTE: Add to the file name of Attack 2 file depending on what you are doing e.g. random-values, same-values,

# Evaluate Clients
for k in range(len(clients)):
    acc = evaluate(clients[k].model, clients[k], mask='test')
    if index in sybil_clients:
        print("Sybil Client{}: {:.2%}".format(clients[k].id, acc))
    else:
        print("Client{}: {:.2%}".format(clients[k].id, acc))

# Evaluate Server
acc = evaluate(server.model, server)
print("Server: {:.2%}".format(acc))
server.g.remove_edges(server.g.edges(form='eid'))
server.g = add_self_loop(server.g)
acc = evaluate(server.model, server)
print("Server: {:.2%}".format(acc))

# Evaluate Test Client
acc = evaluate(server.model, test_client)
print("Test Client{}: {:.2%}".format(test_client.id, acc))
test_client.g.remove_edges(test_client.g.edges(form='eid'))
test_client.g = add_self_loop(test_client.g)
acc = evaluate(server.model, test_client)
print("Test Client{}: {:.2%}".format(test_client.id, acc))

# Save result
torch.save(recorder, './saves/recorder.pt')

# Plot Graph
plotfig(recorder, args)
