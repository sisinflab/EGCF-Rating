from abc import ABC

import torch
import numpy as np
import random


class MFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 random_seed,
                 name="MF",
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

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_normal_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_normal_(self.Gi.weight)
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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, **kwargs):
        user, item = inputs
        gamma_u = torch.squeeze(self.Gu.weight[user]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[item]).to(self.device)

        beta_u = torch.squeeze(self.Bu.weight[user]).to(self.device)
        beta_i = torch.squeeze(self.Bi.weight[item]).to(self.device)

        mu = torch.squeeze(self.Mu).to(self.device)

        rui = torch.sum(gamma_u * gamma_i, 1) + beta_u + beta_i + mu

        return rui

    def predict(self, users, items, **kwargs):
        rui = self.forward(inputs=(users, items))
        return rui

    def train_step(self, batch):
        user, item, r = batch
        rui = self.forward(inputs=(user, item))

        loss = self.loss(torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()
