from dataset import Dataset, StockDataset
from trainer import Trainer
from TSGNN import TSGNN
import config

from torch.utils.data import DataLoader


if __name__ == '__main__':
    stock = Dataset(config)
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = stock.get_dataset()
    train_dataset = StockDataset(X_train, y_train)

    # train
    model = TSGNN(config)
    trainer = Trainer(model, train_dataset, config)
    trainer.train_epoch(DataLoader(train_dataset))

