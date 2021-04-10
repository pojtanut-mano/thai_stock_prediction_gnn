from dataset import Dataset
from trainer import Trainer
from TSGNN import TSGNN
from LSTM import LSTM
from TRS import TRS
from MLP import MLP
from evaluator import Evaluator
import config
import os
import time
import datetime
import pickle
import torch

import pandas as pd


def main():
    os.mkdir(os.path.join(config.checkpoint_dir, config.directory))

    evaluator = Evaluator(config)

    start_time = time.time()
    period_list = list(range(config.train_start, config.full_trade_day - (config.valid_size + config.test_size + config.train_size), config.test_size))

    for period in period_list[config.start_period: config.end_period]:
        i = (period - config.lookback) // config.test_size
        print('Period {}'.format(i+1))
        print('-'*10)

        period_name = 'period_{}'.format(i+1)
        period_dir = os.path.join(config.checkpoint_dir, config.directory, period_name)
        os.mkdir(period_dir)

        best_valid_f1 = 0
        best_valid_loss = 100000000

        # Result dataframe
        result_list = []
        for ind, param in enumerate(config.c_grid):
            print('Search parameters: {}'.format(param))
            stock = Dataset(config, period)
            neighbors = stock.sample_neighbors(param[0], to_tensor=True)
            rel_num = stock.rel_num

            param_grid = {
                "hidden_dims": param[1],
                "optimizer": param[2],
                "weight_decay": param[3],
                "lr": param[4],
                "dropout_rate": param[5]
            }

            if config.model == 'TSGNN':
                model = TSGNN(config, neighbors, rel_num, **param_grid)

            elif config.model == 'LSTM':
                model = LSTM(config, **param_grid)

            elif config.model == 'TRS':
                relation = stock.rel_encoding
                rel_mask = stock.get_relation_mask()
                model = TRS(config, relation, rel_mask, **param_grid)

            else:
                model = MLP(config, **param_grid)

            trainer = Trainer(model, stock, config, evaluator)
            # train
            # Check if the best valid acc is less than curr, if so, set best acc = curr
            if config.mode == 'train' and config.target_type == 'classification':
                if config.model == 'TSGNN':
                    trained_state, valid_f1, score_list, avg_rel_weight = trainer.train_classifier()
                else:
                    trained_state, valid_f1, score_list = trainer.train_classifier()
                result_list.append(param + score_list)
                if best_valid_f1 < valid_f1:
                    print('Better validation f1: {:.4f}'.format(valid_f1))
                    print('-'*20)
                    best_valid_f1 = valid_f1
                    best_state = trained_state
                    best_hist = trainer.get_hist()
                    best_param = param
                    if config.model == 'TSGNN':
                        best_rel_weight = avg_rel_weight

            elif config.mode == 'train' and (config.target_type == 'regression' or config.model == 'TRS'):
                if config.model == 'TSGNN':
                    trained_state, valid_loss, score_list, avg_rel_weight = trainer.train_regressor()
                else:
                    trained_state, valid_loss, score_list = trainer.train_regressor()
                result_list.append(param+score_list)
                if best_valid_loss > valid_loss:
                    print('Better validation loss: {:.4f}'.format(valid_loss))
                    print('-' * 20)
                    best_valid_loss = valid_loss
                    best_state = trained_state
                    best_hist = trainer.get_hist()
                    best_param = param
                    if config.model == 'TSGNN':
                        best_rel_weight = avg_rel_weight

        if config.mode == 'train' and config.target_type == 'classification':
            result_df = pd.DataFrame(result_list, columns=['num_sample_neighbors', 'lstm_hidden_dims', 'optimizer',
                                              'optimizer_weight_decay', 'lr', 'dropout', 'train_acc', 'train_f1', 'valid_acc',
                                              'valid_f1', 'test_acc', 'test_f1'])
        elif config.mode == 'train' and (config.target_type == 'regression' or config.model == 'TRS'):
            result_df = pd.DataFrame(result_list, columns=['num_sample_neighbors', 'lstm_hidden_dims', 'optimizer',
                                              'optimizer_weight_decay', 'lr', 'dropout', 'train_MSE', 'train_MAE', 'valid_MSE', 'valid_MAE', 'test_MSE', 'test_MAE'])

        result_df.to_csv(os.path.join(config.checkpoint_dir, config.directory, period_name, 'grid_search_result.csv'), index=False)
        print('-'*25)
        print('Best param: {}'.format(best_param))
        print('-'*25)

        if config.model == 'TSGNN':
            model = TSGNN(config, neighbors, rel_num, best_param[1], best_param[2], best_param[3], best_param[4], best_param[5])
        elif config.model == 'LSTM':
            model = LSTM(config, best_param[1], best_param[2], best_param[3], best_param[4], best_param[5])
        elif config.model == 'TRS':
            model = TRS(config, relation, rel_mask, best_param[1], best_param[2], best_param[3], best_param[4], best_param[5])
        else:
            model = MLP(config, best_param[1], best_param[2], best_param[3], best_param[4], best_param[5])

        # Load model
        model.load_state(best_state)

        # Save best model
        evaluator.save_model(best_state, os.path.join(config.checkpoint_dir, config.directory, period_name, config.name))

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
            evaluator.export_history(best_hist, period_name)

        # Export each set
        evaluator.export_metrics(stock.get_dataset(), model, result_df, period_name)

        # Export config
        evaluator.save_config(best_param, period_name)

        if config.model == 'TSGNN':
            evaluator.export_rel_weight(best_rel_weight, period_name)

        if i != config.end_period - 1:
            time.sleep(240)

    finish_time = time.time()
    second_used = finish_time - start_time

    print('\nTime used: {}'.format(datetime.timedelta(seconds=second_used)))


if __name__ == '__main__':
    main()