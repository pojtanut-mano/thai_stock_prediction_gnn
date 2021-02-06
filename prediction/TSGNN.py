import torch
from torch import nn, optim
import os


class TSGNN(nn.Module):
    def __init__(self, lr, lstm_output_dims, lstm_input_dims, checkpoint_dir, name,
                 lstm_hidden_dims=64, lstm_layer=1):
        super(TSGNN, self).__init__()
        self.checkpoint_directory = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_directory, name)

        self.lstm = nn.LSTM(lstm_input_dims, lstm_hidden_dims, lstm_layer)