import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import random

EPSILON = 1e-10


class TRS(nn.Module):
    def __init__(self, config, relation, rel_mask, hidden_dims, optimizer, weight_decay, lr):
        super(TRS, self).__init__()
        self.config = config

        self.checkpoint_directory = config.checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_directory, config.directory, config.name)

        # LSTM layer
        if config.lstm_layer > 1:
            self.lstm = nn.LSTM(config.lstm_input_dims, hidden_dims,
                                config.lstm_layer, dropout=config.lstm_dropout)
        else:
            self.lstm = nn.LSTM(config.lstm_input_dims, hidden_dims,
                                config.lstm_layer)

        # Fully connected
        self.explicit_fc = nn.Linear(in_features=relation.shape[2],
                                     out_features=1)
        self.head_fc = nn.Linear(in_features=hidden_dims,
                                 out_features=1)
        self.tail_fc = nn.Linear(in_features=hidden_dims,
                                 out_features=1)

        self.fc1 = nn.Linear(in_features=hidden_dims,
                             out_features=hidden_dims)
        self.fc2 = nn.Linear(in_features=hidden_dims,
                             out_features=1)

        # Optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(params=self.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        else:
            self.optimizer = optim.RMSprop(params=self.parameters(), lr=lr,
                                           weight_decay=weight_decay)

        # Activation
        self.leaky_relu = nn.LeakyReLU()
        self.rel_softmax = nn.Softmax(dim=0)

        # Loss
        self.loss = nn.MSELoss()

        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Current device: {}\n'.format(self.device))
        self.to(self.device)

        # Init weight
        self.apply(self.weight_init)

        # Clip gradient
        if config.clip_grad > 0:
            clip_grad_norm_(self.parameters(), config.clip_grad)

        # lr scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size,
                                gamma=self.config.gamma)

        self.relation = torch.from_numpy(relation).type(torch.float32).to(self.device)
        self.rel_mask = torch.from_numpy(rel_mask).type(torch.float32).to(self.device)

    def forward(self, stock_hist):
        _, (state_embedding, _) = self.lstm(stock_hist)
        state_embedding = torch.squeeze(state_embedding, dim=0)

        relation_importance = torch.squeeze(self.leaky_relu(self.explicit_fc(self.relation)), dim=2)
        ones = torch.ones(self.relation.shape[0], 1).type(torch.float32).to(self.device)

        if self.config.relation_type == 'explicit':
            similarity = torch.matmul(state_embedding, torch.transpose(state_embedding, 0, 1))
            relation_strength = torch.mul(similarity, relation_importance)

        elif self.config.relation_type == 'implicit':
            head_weight = self.leaky_relu(self.head_fc(state_embedding))
            tail_weight = self.leaky_relu(self.tail_fc(state_embedding))

            relation_strength = torch.add(
                torch.add(torch.matmul(head_weight, torch.transpose(ones, 0, 1)),
                          torch.matmul(ones, torch.transpose(tail_weight, 0, 1))),
                relation_importance)

        rel_str_masked = self.rel_softmax(torch.add(self.rel_mask, relation_strength))
        propagated_info = torch.matmul(rel_str_masked, state_embedding)

        # Prediction layer
        fc1_result = self.leaky_relu(self.fc1(propagated_info))
        preds = self.fc2(fc1_result)

        return preds

    def weight_init(self, m):
        """Input: m as a class of pytorch nn Module
           Initialize """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

    def load(self):
        print('... loading from path ...')
        self.load_state_dict(torch.load(self.config.path))

    def init_seed(self):
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)

    def save_model(self):
        print('... saving state ...')
        return self.state_dict()

    def load_state(self, state):
        print('Loading best params...')
        self.load_state_dict(state)
