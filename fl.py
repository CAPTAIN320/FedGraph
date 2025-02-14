import copy
from utils import *
import random
import torch


class Client(object):
    def __init__(self, id, graph, args):
        self.args = args
        self.id = id + 1
        self.g = graph.int().to(args.device)
        self.feats = self.g.ndata['feat']
        self.labels = self.g.ndata['label']
        self.train_mask = self.g.ndata['train_mask']
        self.val_mask = self.g.ndata['val_mask']
        self.test_mask = self.g.ndata['test_mask']
        self.model = init_model(self.args).to(args.device)
        self.optimizer = init_optimizer(self.model, self.args)

    def fork(self, server):
        self.model = copy.deepcopy(server.model)
        self.optimizer = init_optimizer(self.model, self.args)

    def local_update(self):
        for E in range(self.args.E):
            train(self)

class Server(object):
    def __init__(self, graph, args):
        self.args = args
        self.g = graph.int().to(args.device)
        self.feats = self.g.ndata['feat']
        self.labels = self.g.ndata['label']
        self.model = init_model(self.args).to(args.device)
        self.train_mask = self.g.ndata['train_mask']
        self.val_mask = self.g.ndata['val_mask']
        self.test_mask = self.g.ndata['test_mask']
        self.dict = self.model.state_dict()
        self.sybil_clients = []  # List to store sybil clients

    def merge(self, clients):
        weights_zero(self.model)
        all_clients = clients + self.sybil_clients  # Include sybil clients in the training
        clients_states = [copy.deepcopy(client.model.state_dict()) for client in all_clients]
        for key in self.dict.keys():
            for i in range(len(all_clients)):
                self.dict[key] += clients_states[i][key]
            self.dict[key] /= len(all_clients)

