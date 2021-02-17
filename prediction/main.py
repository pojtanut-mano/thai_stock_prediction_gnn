from dataset import Dataset
from trainer import Trainer
from TSGNN import TSGNN
from evaluator import Evaluator
import config


if __name__ == '__main__':
    stock = Dataset(config)
    neighbors = stock.sample_neighbors(to_tensor=True)
    model = TSGNN(config, neighbors, stock.rel_num)
    print(model)
    evaluator = Evaluator(config)
    trainer = Trainer(model, stock, config, evaluator)
    # print(trainer.X_train)

    # train
    trainer.train()
    # print(trainer.X_train.shape)
