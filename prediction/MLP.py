import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import random


class MLP(nn.Module):
    def __init__(self, config, hidden_dims, optimizer, weight_decay, lr, dropout_rate):
        super(MLP, self).__init__()
        self.config = config
        self.checkpoint_directory = config.checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_directory, config.directory, config.name)

        self.hidden_dims = hidden_dims

        # reproducibility
        self.init_seed()

        self.dropout = nn.Dropout(p=dropout_rate)

        # Fully-connected layer
        self.fc_input = nn.Linear(in_features=config.lookback * len(config.feature_list),
                                  out_features=hidden_dims)
        self.fc_hidden_1 = nn.Linear(in_features=hidden_dims, out_features=hidden_dims)
        self.fc_hidden_2 = nn.Linear(in_features=hidden_dims, out_features=hidden_dims)
        self.fc_hidden_3 = nn.Linear(in_features=hidden_dims, out_features=hidden_dims)
        self.fc_hidden_4 = nn.Linear(in_features=hidden_dims, out_features=hidden_dims)

        if config.target_type == 'classification':
            self.fc_output = nn.Linear(in_features=hidden_dims,
                                 out_features=3)
        else:
            self.fc_output = nn.Linear(in_features=hidden_dims,
                                 out_features=1)
        self.softmax = nn.Softmax(dim=1)

        # Utils
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(params=self.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        else:
            self.optimizer = optim.RMSprop(params=self.parameters(), lr=lr,
                                           weight_decay=weight_decay)
        if config.target_type == 'classification':
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

        self.relu = nn.ReLU()

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

    def forward(self, hist):
        # print(hist.shape, end=" ")
        fc_input_r = self.dropout(self.relu(self.fc_input(hist)))
        fc_hidden_1_r = self.dropout(self.relu(self.fc_hidden_1(fc_input_r)))
        fc_hidden_2_r = self.dropout(self.relu(self.fc_hidden_2(fc_hidden_1_r)))
        fc_hidden_3_r = self.dropout(self.relu(self.fc_hidden_3(fc_hidden_2_r)))
        fc_hidden_4_r = self.dropout(self.relu(self.fc_hidden_4(fc_hidden_3_r)))
        fc_output_r = self.dropout(self.relu(self.fc_output(fc_hidden_4_r)))
        # print(fc_output_r.shape)
        preds = self.softmax(fc_output_r)

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
