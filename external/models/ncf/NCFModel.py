from abc import ABC

import torch
import numpy as np
import random
from collections import OrderedDict


class NCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 dropout,
                 dense_size,
                 random_seed,
                 name="NCF",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.dense_layer_size = [self.embed_k * 2] + dense_size

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_normal_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_normal_(self.Gi.weight)
        self.Gi.to(self.device)

        # Dense part
        dense_network_list = []
        for idx, _ in enumerate(self.dense_layer_size[:-1]):
            dense_network_list.append(('dense_' + str(idx), torch.nn.Linear(in_features=self.dense_layer_size[idx],
                                                                            out_features=self.dense_layer_size[
                                                                                idx + 1],
                                                                            bias=True)))
            dense_network_list.append(('relu_' + str(idx), torch.nn.ReLU()))
            if self.dropout > 0.0:
                dense_network_list.append(('dropout_' + str(idx), torch.nn.Dropout(p=self.dropout)))
        dense_network_list.append(('out', torch.nn.Linear(in_features=self.dense_layer_size[-1],
                                                          out_features=1,
                                                          bias=True)))
        self.dense_network = torch.nn.Sequential(OrderedDict(dense_network_list))
        self.dense_network.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        rui = self.dense_network(torch.concat([gamma_u, gamma_i], dim=1))

        return rui

    def predict(self, users, items, **kwargs):
        rui = self.dense_network(torch.concat([self.Gu.weight[users], self.Gi.weight[items]], dim=1))
        return rui

    def train_step(self, batch):
        user, item, r = batch
        rui = self.forward(inputs=(self.Gu.weight[user], self.Gi.weight[item]))

        loss = self.loss(torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()
