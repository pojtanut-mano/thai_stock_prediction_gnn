import torch
from torch import nn, optim
import os


class TSGNN(nn.Module):
    def __init__(self, config):
        super(TSGNN, self).__init__()
        self.config = config
        self.checkpoint_directory = config.checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_directory, config.name)

        self.lstm = nn.LSTM(config.lstm_input_dims, config.lstm_hidden_dims,
                            config.lstm_layer)

        # Utils
        self.optimizer = optim.Adam(params=self.parameters(), lr=config.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Current device: {}\n'.format(self.device))
        self.to(self.device)

    def forward(self, stock_hist):
        _, (lstm_embedding, _) = self.lstm(stock_hist)
        lstm_embedding = torch.squeeze(lstm_embedding, dim=0)
        return lstm_embedding

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
