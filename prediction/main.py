from dataset import Dataset
from trainer import Trainer
from TSGNN import TSGNN
from evaluator import Evaluator
import config

import pandas as pd


if __name__ == '__main__':
    stock = Dataset(config)
    neighbors = stock.sample_neighbors(to_tensor=True)
    model = TSGNN(config, neighbors, stock.rel_num)
    print(model)
    evaluator = Evaluator(config)
    trainer = Trainer(model, stock, config, evaluator)

    # train
    trainer.train()

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
    trained_model = trainer.get_model()

    # Export history graph
    history = trainer.get_hist()
    evaluator.export_history(history)

    # Export each set
    evaluator.export_metrics(stock.get_dataset(), trained_model, result_df)
