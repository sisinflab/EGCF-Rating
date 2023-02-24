from abc import ABC

from torch_geometric.nn import LGConv, GATConv
from collections import OrderedDict

import torch
import torch_geometric
import numpy as np
import random

from torch_sparse import SparseTensor


class RoGERModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 l_ind,
                 embed_k,
                 n_layers,
                 edge_features,
                 edge_index,
                 lm,
                 eps,
                 aggr,
                 dense,
                 random_seed,
                 name="RoGER",
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
        self.l_ind = l_ind
        self.n_layers = n_layers

        self.L0 = edge_index[2].clone().to(self.device)
        self.edge_index = edge_index.to(self.device)
        self.eps = eps

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        self.Bu = torch.nn.Embedding(self.num_users, 1)
        torch.nn.init.xavier_normal_(self.Bu.weight)
        self.Bu.to(self.device)
        self.Bi = torch.nn.Embedding(self.num_items, 1)
        torch.nn.init.xavier_normal_(self.Bi.weight)
        self.Bi.to(self.device)

        self.Mu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((1, 1))))
        self.Mu.to(self.device)

        self.lm = lm
        self.aggr = aggr

        self.edge_embeddings_interactions = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
        self.feature_dim = edge_features.shape[1]

        # create node-node textual
        propagation_node_node_textual_list = []
        for _ in range(self.n_layers):
            propagation_node_node_textual_list.append(
                (LGConv(normalize=True), 'x, edge_index -> x'))

        self.node_node_textual_network = torch_geometric.nn.Sequential('x, edge_index',
                                                                       propagation_node_node_textual_list)
        self.node_node_textual_network.to(self.device)

        if self.aggr == 'sim':
            # projection user
            self.projection_user = torch.nn.Linear(self.edge_embeddings_interactions.shape[-1], self.embed_k)
            self.projection_user.to(self.device)

            # projection item
            self.projection_item = torch.nn.Linear(self.edge_embeddings_interactions.shape[-1], self.embed_k)
            self.projection_item.to(self.device)

        elif self.aggr == 'nn':
            self.dense_layer_size = [self.embed_k * 2 + self.feature_dim] + dense + [1]
            self.num_dense_layers = len(self.dense_layer_size)
            dense_network_list_user = []
            for idx, _ in enumerate(self.dense_layer_size[:-1]):
                dense_network_list_user.append(
                    ('dense_' + str(idx), torch.nn.Linear(in_features=self.dense_layer_size[idx],
                                                          out_features=self.dense_layer_size[
                                                              idx + 1],
                                                          bias=False)))
                dense_network_list_user.append(('relu_' + str(idx), torch.nn.ReLU()))
            self.dense_network_user = torch.nn.Sequential(OrderedDict(dense_network_list_user))
            self.dense_network_user.to(self.device)
            dense_network_list_item = []
            for idx, _ in enumerate(self.dense_layer_size[:-1]):
                dense_network_list_item.append(
                    ('dense_' + str(idx), torch.nn.Linear(in_features=self.dense_layer_size[idx],
                                                          out_features=self.dense_layer_size[
                                                              idx + 1],
                                                          bias=False)))
                dense_network_list_item.append(('relu_' + str(idx), torch.nn.ReLU()))
            self.dense_network_item = torch.nn.Sequential(OrderedDict(dense_network_list_item))
            self.dense_network_item.to(self.device)
        else:
            self.attention_user = GATConv(in_channels=self.embed_k,
                                          out_channels=self.embed_k,
                                          concat=True,
                                          edge_dim=self.feature_dim,
                                          add_self_loops=False,
                                          bias=False)
            self.attention_user.to(self.device)
            self.attention_item = GATConv(in_channels=self.embed_k,
                                          out_channels=self.embed_k,
                                          concat=True,
                                          edge_dim=self.feature_dim,
                                          add_self_loops=False,
                                          bias=False)
            self.attention_item.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = torch.nn.MSELoss()

    def propagate_embeddings(self, evaluate=False):
        all_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        for layer in range(self.n_layers):
            if evaluate:
                with torch.no_grad():
                    updates = self.update_adjacency(all_embeddings)
                    final_values = self.lm * self.L0.to(self.device) + (1 - self.lm) * updates
                    self.edge_index = torch.stack([self.edge_index[0], self.edge_index[1], final_values], dim=0)
                    all_embeddings = list(
                        self.node_node_textual_network.children()
                    )[layer](all_embeddings.to(self.device),
                             self.edge_index_to_adj(self.edge_index).to(self.device))
            else:
                with torch.no_grad():
                    updates = self.update_adjacency(all_embeddings)
                    final_values = self.lm * self.L0.to(self.device) + (1 - self.lm) * updates
                    self.edge_index = torch.stack([self.edge_index[0], self.edge_index[1], final_values], dim=0)
                all_embeddings = list(
                    self.node_node_textual_network.children()
                )[layer](all_embeddings.to(self.device),
                         self.edge_index_to_adj(self.edge_index).to(self.device))

        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi

    def edge_index_to_adj(self, edge_index, eps=True):
        rows = edge_index[0].long().to(self.device)
        cols = edge_index[1].long().to(self.device)
        values = edge_index[2].float().to(self.device)

        if eps:
            new_indices = torch.where(values >= self.eps)[0]
            return SparseTensor(row=rows[new_indices],
                                col=cols[new_indices],
                                value=torch.full(size=(new_indices.shape[0],),
                                                 fill_value=1.0, dtype=torch.float32, device=self.device),
                                sparse_sizes=(self.num_users + self.num_items,
                                              self.num_users + self.num_items))
        else:
            return SparseTensor(row=rows,
                                col=cols,
                                value=values,
                                sparse_sizes=(self.num_users + self.num_items,
                                              self.num_users + self.num_items))

    def update_adjacency(self, node_embeddings):
        row, col, _ = self.edge_index
        row, col = row.long(), col.long()
        row_nodes = node_embeddings[row[:row.shape[0] // 2]]
        col_nodes = node_embeddings[row[:col.shape[0] // 2] - self.num_users]

        if self.aggr == 'sim':
            user_item = torch.relu(torch.nn.functional.cosine_similarity(
                torch.mul(row_nodes, self.projection_user(self.edge_embeddings_interactions)),
                torch.mul(col_nodes, self.projection_user(self.edge_embeddings_interactions))
            ))
            item_user = torch.relu(torch.nn.functional.cosine_similarity(
                torch.mul(col_nodes, self.projection_item(self.edge_embeddings_interactions)),
                torch.mul(row_nodes, self.projection_item(self.edge_embeddings_interactions))
            ))
            updates = torch.concat([user_item, item_user], dim=0)
            return updates

        elif self.aggr == 'nn':
            user_item = torch.squeeze(self.dense_network_user(torch.concat(
                [row_nodes, self.edge_embeddings_interactions, col_nodes], dim=-1)))
            item_user = torch.squeeze(self.dense_network_item(torch.concat(
                [col_nodes, self.edge_embeddings_interactions, row_nodes], dim=-1)))
            updates = torch.concat([user_item, item_user], dim=0)
            return updates

        else:
            _, user_item = self.attention_user(
                node_embeddings,
                self.edge_index_to_adj(self.edge_index[:, :self.edge_index.shape[1] // 2], eps=False),
                self.edge_embeddings_interactions,
                return_attention_weights=True)
            _, item_user = self.attention_item(
                node_embeddings,
                self.edge_index_to_adj(self.edge_index[:, self.edge_index.shape[1] // 2:], eps=False),
                self.edge_embeddings_interactions,
                return_attention_weights=True)
            user_item = torch.squeeze(user_item.coo()[2])
            item_user = torch.squeeze(item_user.coo()[2])
            updates = torch.concat([user_item, item_user], dim=0)
            return updates

    def forward(self, inputs, **kwargs):
        gu, gi, bu, bi = inputs

        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        beta_u = torch.squeeze(bu).to(self.device)
        beta_i = torch.squeeze(bi).to(self.device)

        mu = torch.squeeze(self.Mu).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1) + beta_u + beta_i + mu

        return xui

    def predict(self, gu, gi, users, items, **kwargs):
        rui = self.forward(inputs=(gu, gi,
                                   self.Bu.weight[users], self.Bi.weight[items]))
        return rui

    @staticmethod
    def get_loss_ind(x1, x2):
        # reference: https://recbole.io/docs/_modules/recbole/model/general_recommender/dgcf.html
        def _create_centered_distance(x):
            r = torch.sum(x * x, dim=1, keepdim=True)
            v = r - 2 * torch.mm(x, x.T + r.T)
            z_v = torch.zeros_like(v)
            v = torch.where(v > 0.0, v, z_v)
            D = torch.sqrt(v + 1e-8)
            D = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) + torch.mean(D)
            return D

        def _create_distance_covariance(d1, d2):
            v = torch.sum(d1 * d2) / (d1.shape[0] * d1.shape[0])
            z_v = torch.zeros_like(v)
            v = torch.where(v > 0.0, v, z_v)
            dcov = torch.sqrt(v + 1e-8)
            return dcov

        loss_ind = 0
        for xx1, xx2 in zip(x1, x2):
            D1 = _create_centered_distance(xx1)
            D2 = _create_centered_distance(xx2)

            dcov_12 = _create_distance_covariance(D1, D2)
            dcov_11 = _create_distance_covariance(D1, D1)
            dcov_22 = _create_distance_covariance(D2, D2)

            # calculate the distance correlation
            value = dcov_11 * dcov_22
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            loss_ind += dcov_12 / (torch.sqrt(value) + 1e-10)
        return loss_ind

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, item, r = batch
        rui = self.forward(inputs=(gu[user], gi[item],
                                   self.Bu.weight[user], self.Bi.weight[item]))

        loss = self.loss(torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float))

        if self.aggr == 'sim':
            loss_ind = self.l_ind * self.get_loss_ind([self.projection_user.weight],
                                                      [self.projection_item.weight])

        elif self.aggr == 'nn':
            loss_ind = self.l_ind * self.get_loss_ind([*self.dense_network_user.parameters()],
                                                      [*self.dense_network_item.parameters()])

        else:
            loss_ind = self.l_ind * self.get_loss_ind([torch.squeeze(w, dim=0) for w in self.attention_user.parameters()],
                                                      [torch.squeeze(w, dim=0) for w in self.attention_item.parameters()])

        loss += loss_ind

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()
