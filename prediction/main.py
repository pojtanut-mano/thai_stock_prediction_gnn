from dataset import Dataset
from trainer import Trainer
from TSGNN import TSGNN
from LSTM import LSTM
from evaluator import Evaluator
import config
import os

import pandas as pd


def main():
    os.mkdir(os.path.join(config.checkpoint_dir, config.directory))

    with open(os.path.join(config.checkpoint_dir, config.directory, 'config.txt'), 'w') as f:
        f.write(" ".join(
            [str(config.lstm_hidden_dims), str(config.optimizer), str(config.optimizer_weight_decay), str(config.lr)]))

    stock = Dataset(config)
    neighbors = stock.sample_neighbors(to_tensor=True)
    if config.model == 'TSGNN':
        model = TSGNN(config, neighbors, stock.rel_num)
    else:
        model = LSTM(config)
    print(model)
    evaluator = Evaluator(config)
    trainer = Trainer(model, stock, config, evaluator)

    # train
    if config.mode == 'train' and config.target_type == 'classification':
        trainer.train_classifier()
    elif config.mode == 'train' and config.target_type == 'regression':
        trainer.train_regressor()
    elif config.mode == 'test':
        model.load()

    # Retrieve model for printing out the result
    train_df = pd.DataFrame(columns=stock.tickers)
    valid_df = pd.DataFrame(columns=stock.tickers)
    test_df = pd.DataFrame(columns=stock.tickers)

    train_df['Date'] = stock.train_date
    valid_df['Date'] = stock.valid_date
    test_df['Date'] = stock.test_date

    train_df = train_df.set_index('Date')
    valid_df = valid_df.set_index('Date')
    test_df = test_df.set_index('Date')

    result_df = (train_df, valid_df, test_df)
    if config.mode == 'train':
        model.load_checkpoint()

    # Export history graph
    history = trainer.get_hist()
    if config.mode == 'train':
        evaluator.export_history(history)

    # Export each set
    evaluator.export_metrics(stock.get_dataset(), model, result_df)


if __name__ == '__main__':
    main()
