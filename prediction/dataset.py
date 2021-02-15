import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import pickle


class Dataset:
    def __init__(self, config):
        self.config = config
        self.mkt_dir = config.market_directory
        self.rel_dir = config.relation_directory
        self.feature_list = config.feature_list
        self.lookback = config.lookback

        self.train_set, self.test_set, self.valid_set = [], [], []
        self.train_label, self.test_label, self.valid_label = [], [], []

        self.load()

    def load(self):
        with open(os.path.join(self.rel_dir, 'ordered_tickers.pkl'), 'rb') as f:
            ordered_tickers = pickle.load(f)

        print(ordered_tickers[:5])
        train_target_start_idx = self.lookback
        valid_start_idx = self.lookback + self.config.train_size
        test_start_idx = valid_start_idx + self.config.valid_size

        # Scale
        if self.config.scale_type == 'MinMax':
            print("Initialize MinMax scaling")
            self.min = []
            self.max = []
        elif self.config.scale_type == 'normalize':
            print("Initialize normalization scaling")
            self.mean = []
            self.std = []

        for ticker in ordered_tickers:
            df = pd.read_csv(os.path.join(self.mkt_dir, '{}.csv'.format(ticker)))
            df = df[self.feature_list]
            if self.config.scale_type == 'MinMax':
                df, min_, max_ = self.minmax_scaler(df)
                self.min.append('{}'.format(min_))
                self.max.append('{}'.format(max_))
            elif self.config.scale_type == 'normalize':
                df, mean_, std_ = self.normalize(df)
                self.mean.append('{}'.format(mean_))
                self.std.append('{}'.format(std_))

            self.train_set.append(df.iloc[:valid_start_idx-1].values)
            self.train_label.append(df.iloc[train_target_start_idx:valid_start_idx].values)

            self.valid_set.append(df.iloc[valid_start_idx-self.lookback:test_start_idx-1].values)
            self.valid_label.append(df.iloc[valid_start_idx: test_start_idx].values)

            self.test_set.append(df.iloc[test_start_idx-self.lookback:test_start_idx+self.config.test_size-1].values)
            self.test_label.append(df.iloc[test_start_idx:test_start_idx+self.config.test_size].values)

        # if self.config.scale_type == 'MinMax':
        #     self.params_mem = pd.DataFrame({'tickers': ordered_tickers, 'min_value': self.min,
        #                                     'max_value': self.max})


        # Convert train set, valid set, test set to
        # shape: (number of samples, number of companies, lookback, feature(s))

        self.train_set = self.create_batch(self.config.train_size, self.train_set)
        self.valid_set = self.create_batch(self.config.valid_size, self.valid_set)
        self.test_set = self.create_batch(self.config.test_size, self.test_set)
        self.train_set = np.swapaxes(self.train_set, 1, 2)
        self.valid_set = np.swapaxes(self.valid_set, 1, 2)
        self.test_set = np.swapaxes(self.test_set, 1, 2)
        if self.config.verbose >= 1:
            print("training set shape: {}\nvalid set shape: {}\ntest set shape: {}\n".format(self.train_set.shape,
                                                                                             self.valid_set.shape,
                                                                                             self.test_set.shape))

        self.train_label = self.create_label_batch(self.config.train_size, self.train_label)
        self.valid_label = self.create_label_batch(self.config.valid_size, self.valid_label)
        self.test_label = self.create_label_batch(self.config.test_size, self.test_label)
        if self.config.verbose >= 1:
            print("training label shape: {}\nvalid label shape: {}\ntest label shape: {}\n".format(self.train_label.shape,
                                                                                                 self.valid_label.shape,
                                                                                                 self.test_label.shape))
        # Convert pytorch
        # self.create_torch_dataset()

    def create_batch(self, size, data):
        # create new training set
        moving_window = []
        for i in range(size):
            one_step_moving_window = []
            for one_stock in data:
                one_step_moving_window.append(np.array(one_stock[i:i+self.lookback]))
            one_step_moving_window = np.stack(one_step_moving_window, axis=0)
            moving_window.append(one_step_moving_window)

        moving_window = np.stack(moving_window, axis=0)
        return moving_window

    def create_label_batch(self, size, labels):
        moving_label = []
        for i in range(size):
            temp_label_list = []
            for label in labels:
                temp_label_list.append(label[i])
            temp_label_list = np.stack(temp_label_list, axis=0)
            moving_label.append(temp_label_list)

        moving_label = np.stack(moving_label, axis=0)
        return moving_label

    # def create_torch_dataset(self):
    #     self.torch_train_set, self.torch_test_set, self.torch_valid_set = torch.from_numpy(self.train_set), \
    #                                                                       torch.from_numpy(self.test_set),\
    #                                                                       torch.from_numpy(self.valid_set)
    #     self.torch_train_label, self.torch_test_label, self.torch_valid_label = torch.from_numpy(self.train_label), \
    #                                                                             torch.from_numpy(self.test_label), \
    #                                                                             torch.from_numpy(self.valid_label)

    # Utils
    def minmax_scaler(self, df):
        min_ = df.min()
        max_ = df.max()
        return (df - min_) / (max_ - min_), min_.values.tolist(), max_.values.tolist()

    def normalize(self, df):
        mean_ = df.mean()
        std_ = df.std()
        return (df - mean_) / std_, mean_.values.tolist(), std_.values.tolist()

    # Return values
    def get_dataset(self):
        return (self.train_set, self.train_label), (self.test_set, self.test_label),\
                    (self.valid_set, self.valid_label)

    # debugging method
    def print_sample_batch(self):
        print('training examples of first company from lag 0 and lag 1')
        print(self.train_set[0][0], '\n\n', self.train_set[1][0])
        print('Target of lag 0 and lag1')
        print(self.train_label[0][0], self.train_label[1][0])


class StockDataset(Dataset):
    """Stock dataset."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx], self.labels[idx]
