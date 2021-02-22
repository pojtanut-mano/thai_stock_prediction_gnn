import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os

EPSILON = 1e-10


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.checkpoint_directory = config.checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_directory, config.directory, config.name)

        # reproducibility
        self.init_seed()

        if config.lstm_layer > 1:
            self.lstm = nn.LSTM(config.lstm_input_dims, config.lstm_hidden_dims,
                                config.lstm_layer, dropout=config.lstm_dropout)
        else:
            self.lstm = nn.LSTM(config.lstm_input_dims, config.lstm_hidden_dims,
                                config.lstm_layer)

        # Fully-connected layer
        if config.target_type == 'classification':
            self.fc1 = nn.Linear(in_features=config.lstm_hidden_dims,
                                 out_features=3)
        else:
            self.fc1 = nn.Linear(in_features=config.lstm_hidden_dims,
                                 out_features=1)
        self.softmax = nn.Softmax(dim=1)

        # Utils
        if config.optimizer == 'Adam':
            self.optimizer = optim.Adam(params=self.parameters(), lr=config.lr,
                                        weight_decay=config.optimizer_weight_decay)
        else:
            self.optimizer = optim.RMSprop(params=self.parameters(), lr=config.lr,
                                           weight_decay=config.optimizer_weight_decay)
        if config.target_type == 'classification':
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

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
        state_embedding = torch.squeeze(state_embedding, dim=0)

        # Prediction layer
        preds = self.fc1(state_embedding)
        if self.config.target_type == 'classification':
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
