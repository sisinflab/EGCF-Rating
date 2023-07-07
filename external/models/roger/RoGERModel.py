from abc import ABC

from torch_geometric.nn import GCNConv, GATConv
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
                 embed_k,
                 n_layers,
                 edge_features,
                 edge_index,
                 lm,
                 alpha,
                 beta,
                 gamma,
                 iter_GSL,
                 eps_adj,
                 aggr,
                 drop,
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
        self.n_layers = n_layers
        self.L0 = torch.ones((edge_index.shape[1],), dtype=torch.float32, device=self.device)
        self.edge_index = edge_index.to(self.device)

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
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.aggr = aggr
        self.drop = drop
        self.iter_GSL = iter_GSL
        self.eps_adj = eps_adj

        self.edge_embeddings_interactions = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
        self.feature_dim = edge_features.shape[1]

        # create node-node textual
        propagation_node_node_textual_list = []
        for _ in range(self.n_layers):
            propagation_node_node_textual_list.append(
                (GCNConv(in_channels=self.embed_k,
                         out_channels=self.embed_k,
                         normalize=True,
                         add_self_loops=False,
                         bias=True), 'x, edge_index -> x'))

        self.node_node_textual_network = torch_geometric.nn.Sequential('x, edge_index',
                                                                       propagation_node_node_textual_list)
        self.node_node_textual_network.to(self.device)

        if self.aggr == 'sim':
            # projection
            self.projection = torch.nn.Linear(self.edge_embeddings_interactions.shape[-1], self.embed_k)
            self.projection.to(self.device)

        elif self.aggr == 'nn':
            self.dense_layer_size = [self.embed_k * 2 + self.feature_dim] + dense
            self.num_dense_layers = len(self.dense_layer_size)
            dense_network_list = []
            for idx, _ in enumerate(self.dense_layer_size[:-1]):
                dense_network_list.append(
                    ('dense_' + str(idx), torch.nn.Linear(in_features=self.dense_layer_size[idx],
                                                          out_features=self.dense_layer_size[
                                                              idx + 1],
                                                          bias=True)))
                dense_network_list.append(('drop_' + str(idx), torch.nn.Dropout(p=self.drop)))
                dense_network_list.append(('relu_' + str(idx), torch.nn.ReLU()))
            dense_network_list.append(('out', torch.nn.Linear(in_features=self.dense_layer_size[-1],
                                                              out_features=1,
                                                              bias=True)))
            dense_network_list.append(('relu', torch.nn.ReLU()))
            self.dense_network = torch.nn.Sequential(OrderedDict(dense_network_list))
            self.dense_network.to(self.device)
        else:
            self.attention = GATConv(in_channels=self.embed_k,
                                     out_channels=self.embed_k,
                                     concat=True,
                                     edge_dim=self.feature_dim,
                                     add_self_loops=False,
                                     bias=True)
            self.attention.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = torch.nn.MSELoss()

    def propagate_embeddings(self, evaluate=False):
        all_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        for layer in range(self.n_layers):
            if evaluate:
                if self.aggr == 'nn':
                    self.dense_network.eval()
                with torch.no_grad():
                    updates = self.update_adjacency(all_embeddings)
                    curr_raw_adj = self.edge_index_to_adj(
                        torch.stack([self.edge_index[0], self.edge_index[1], updates], dim=0))
                    final_values = self.lm * self.L0.to(self.device) + (1 - self.lm) * updates
                    edge_index = torch.stack([self.edge_index[0], self.edge_index[1], final_values], dim=0)
                    all_embeddings = torch.relu(list(
                        self.node_node_textual_network.children()
                    )[layer](all_embeddings.to(self.device),
                             self.edge_index_to_adj(edge_index).to(self.device)))
            else:
                updates = self.update_adjacency(all_embeddings)
                curr_raw_adj = self.edge_index_to_adj(
                    torch.stack([self.edge_index[0], self.edge_index[1], updates], dim=0))
                final_values = self.lm * self.L0.to(self.device) + (1 - self.lm) * updates
                edge_index = torch.stack([self.edge_index[0], self.edge_index[1], final_values], dim=0)
                all_embeddings = torch.relu(list(
                    self.node_node_textual_network.children()
                )[layer](torch.dropout(all_embeddings.to(self.device), p=self.drop, train=not evaluate),
                         self.edge_index_to_adj(edge_index).to(self.device)))

        if evaluate:
            if self.aggr == 'nn':
                self.dense_network.train()

        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi, curr_raw_adj

    def edge_index_to_adj(self, edge_index):
        rows = edge_index[0].long().to(self.device)
        cols = edge_index[1].long().to(self.device)
        values = edge_index[2].float().to(self.device)

        return SparseTensor(row=rows,
                            col=cols,
                            value=values,
                            sparse_sizes=(self.num_users + self.num_items,
                                          self.num_users + self.num_items))

    def update_adjacency(self, node_embeddings):
        row, col = self.edge_index
        row, col = row.long(), col.long()
        row_nodes = node_embeddings[row[:row.shape[0] // 2]]
        col_nodes = node_embeddings[row[:col.shape[0] // 2] - self.num_users]
        if self.aggr == 'sim':
            user_item = torch.relu(torch.nn.functional.cosine_similarity(
                torch.mul(row_nodes, self.projection(self.edge_embeddings_interactions)),
                torch.mul(col_nodes, self.projection(self.edge_embeddings_interactions))
            ))
            updates = torch.concat([user_item, user_item], dim=0)
            return updates

        elif self.aggr == 'nn':
            user_item = torch.squeeze(self.dense_network(torch.concat(
                [row_nodes, self.edge_embeddings_interactions, col_nodes], dim=-1)))
            updates = torch.concat([user_item, user_item], dim=0)
            return updates

        else:
            edge_index = self.edge_index[:, :self.edge_index.shape[1] // 2].clone()
            edge_index = torch.concat(
                [edge_index.to(self.device), torch.ones((1, edge_index.shape[1]), device=self.device)])
            _, user_item = self.attention(
                node_embeddings,
                self.edge_index_to_adj(edge_index),
                self.edge_embeddings_interactions,
                return_attention_weights=True)
            user_item = torch.squeeze(user_item.coo()[2])
            updates = torch.concat([user_item, user_item], dim=0)
            return updates

    def add_graph_loss(self, curr_raw_adj, gu, gi):
        # computes graph loss
        graph_loss = 0
        features = torch.cat((gu.to(self.device), gi.to(self.device)), 0)
        curr_raw_adj = curr_raw_adj.to_dense()
        L = torch.diagflat(torch.sum(curr_raw_adj, -1)) - curr_raw_adj
        # SMOOTHNESS LOSS (decreasing alpha, increases smoothness)
        graph_loss += self.alpha * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(
            np.prod(curr_raw_adj.shape))
        ones_vec = torch.ones(curr_raw_adj.size(-1))
        # DEGREE LOSS (HIGHER BETA GIVES NEGATIVE CONTRIBUTE)
        graph_loss += -self.beta * torch.mm(ones_vec.unsqueeze(0), torch.log(
            torch.mm(curr_raw_adj.to(self.device), ones_vec.unsqueeze(-1).to(self.device) + 1e-12)).squeeze() / curr_raw_adj.shape[-1]
        # SPARSITY LOSS
        graph_loss += self.gamma * torch.sum(torch.pow(curr_raw_adj, 2)) / int(np.prod(curr_raw_adj.shape))
        return graph_loss

    def SquaredFrobeniusNorm(self, X):
        return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))

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

    def diff(self, X, Y, Z):
        X = X.to_dense()
        Y = Y.to_dense()
        Z = Z.to_dense()
        assert X.shape == Y.shape
        diff_ = torch.sum(torch.pow(X - Y, 2))
        norm_ = torch.sum(torch.pow(Z, 2))
        diff_ = diff_ / torch.clamp(norm_, min=1e-12)
        return diff_

    def train_step(self, batch):
        all_losses = torch.zeros(self.iter_GSL)
        iter = 0
        gu, gi, curr_raw_adj = self.propagate_embeddings()
        user, item, r = batch
        rui = self.forward(inputs=(gu[user], gi[item],
                                   self.Bu.weight[user], self.Bi.weight[item]))
        loss = self.loss(torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float))
        graph_loss = self.add_graph_loss(curr_raw_adj, gu, gi)
        loss += graph_loss
        all_losses[iter] = loss
        pre_raw_adj = curr_raw_adj
        first_raw_adj = curr_raw_adj
        while (iter == 0 or self.diff(curr_raw_adj, pre_raw_adj, first_raw_adj).item() > self.eps_adj) and iter < self.iter_GSL - 1:
            iter += 1
            pre_raw_adj = curr_raw_adj
            gu, gi, curr_raw_adj = self.propagate_embeddings()
            user, item, r = batch
            rui = self.forward(inputs=(gu[user], gi[item],
                                       self.Bu.weight[user], self.Bi.weight[item]))
            loss = self.loss(torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float))
            graph_loss = self.add_graph_loss(curr_raw_adj, gu, gi)
            loss += graph_loss
            all_losses[iter] = loss
        loss = all_losses[0] + torch.mean(all_losses[1:])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()
