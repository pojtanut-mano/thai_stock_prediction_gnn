import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import random


class LSTM(nn.Module):
    def __init__(self, config, hidden_dims, optimizer, weight_decay, lr, dropout_rate):
        super(LSTM, self).__init__()
        self.config = config
        self.checkpoint_directory = config.checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_directory, config.directory, config.name)

        # reproducibility
        self.init_seed()

        self.lstm = nn.LSTM(config.lstm_input_dims, hidden_dims,
                            config.lstm_layer)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Fully-connected layer
        self.fc = nn.Linear(in_features=hidden_dims,
                            out_features=hidden_dims)
        self.fc1 = nn.Linear(in_features=hidden_dims,
                             out_features=3)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        # Utils
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(params=self.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        else:
            self.optimizer = optim.RMSprop(params=self.parameters(), lr=lr,
                                           weight_decay=weight_decay)
        self.loss = nn.CrossEntropyLoss()

        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Current device: {}\n'.format(self.device))
        self.to(self.device)

        # Weight init
        self.apply(self.weight_init)

        # Clip gradient
        if config.clip_grad > 0:
            clip_grad_norm_(self.parameters(), config.clip_grad)

        # lr scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size,
                                gamma=self.config.gamma)

    def forward(self, stock_hist):
        _, (state_embedding, _) = self.lstm(stock_hist)
        state_embedding = self.dropout(state_embedding)
        state_embedding = torch.squeeze(state_embedding, dim=0)

        # Prediction layer
        # fc_result = self.leaky_relu(self.fc(state_embedding))
        preds = self.dropout(self.fc1(state_embedding))
        preds = self.softmax(preds)

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
