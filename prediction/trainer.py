import torch
from torch.utils.data import DataLoader
from dataset import StockDataset


class Trainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def train(self):
        loader = DataLoader(self.data)
        for epoch in self.range(self.config.epochs):
            loss, metrics = self.train_epoch(loader)

    def train_epoch(self, dataloader):
        for input_hist, label in dataloader:
            # Squeeze one dim
            input_hist = torch.squeeze(input_hist, dim=0).type(torch.float32).to(self.model.device)
            label = torch.squeeze(label, dim=0).type(torch.float32).to(self.model.device)
            # Train mode
            self.model.train()
            self.train_step(input_hist, label)

    def train_step(self, x, y):
        embedding = self.model.forward(x)
        print(embedding.shape)