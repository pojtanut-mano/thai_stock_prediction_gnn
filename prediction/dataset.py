import pandas as pd
import numpy as np
import torch

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
        print('Loading dataset...\n')
        # Load tickers
        with open(os.path.join(self.rel_dir, 'ordered_tickers.pkl'), 'rb') as f:
            ordered_tickers = pickle.load(f)

        # Load adjacency matrices for each relation type
        self.rel_encoding = np.load(os.path.join(self.rel_dir, self.config.adjacency_matrix_path_name))
        self.rel_encoding = np.swapaxes(self.rel_encoding, 0, 2)

        new_rel_encoding = []
        for relation in self.rel_encoding:
            np.fill_diagonal(relation, 0)
            new_rel_encoding.append(relation)
        self.rel_encoding = np.stack(new_rel_encoding, axis=0)

        self.neighbors = []
        for adj_mat in self.rel_encoding:
            rel_neighbors = []
            for row in adj_mat:
                rel_neighbors.append(row.nonzero()[0] + 1)
            self.neighbors.append(rel_neighbors)

        self.rel_num = self.rel_encoding.sum(axis=2)

        print('Adjacency matrix shape: {}\n'.format(self.rel_encoding.shape))

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
            raw_df = pd.read_csv(os.path.join(self.mkt_dir, '{}.csv'.format(ticker)))
            df = raw_df[self.feature_list]
            if self.config.target_type == 'classification':  # 0 = down, 1 = neutral, 2 = up
                self.train_label.append(df.iloc[train_target_start_idx:valid_start_idx, 0].apply(self.classify).values)
                self.valid_label.append(df.iloc[valid_start_idx: test_start_idx, 0].apply(self.classify).values)
                self.test_label.append(
                    df.iloc[test_start_idx:test_start_idx + self.config.test_size, 0].apply(self.classify).values)

            else:
                self.train_label.append(df.iloc[train_target_start_idx:valid_start_idx].values)
                self.valid_label.append(df.iloc[valid_start_idx: test_start_idx].values)
                self.test_label.append(df.iloc[test_start_idx:test_start_idx + self.config.test_size].values)

            if self.config.scale_type == 'MinMax':
                df, min_, max_ = self.minmax_scaler(df)
                self.min.append('{}'.format(min_))
                self.max.append('{}'.format(max_))
            elif self.config.scale_type == 'normalize':
                df, mean_, std_ = self.normalize(df)
                self.mean.append('{}'.format(mean_))
                self.std.append('{}'.format(std_))

            # Append dataset
            self.train_set.append(df.iloc[:valid_start_idx - 1].values)
            self.valid_set.append(df.iloc[valid_start_idx - self.lookback:test_start_idx - 1].values)
            self.test_set.append(df.iloc[test_start_idx - self.lookback:test_start_idx + self.config.test_size - 1].values)

        # Save date for exporting
        date = raw_df['Date']
        self.train_date = date[train_target_start_idx:valid_start_idx].reset_index(drop=True)
        self.valid_date = date[valid_start_idx: test_start_idx].reset_index(drop=True)
        self.test_date = date[test_start_idx:test_start_idx + self.config.test_size].reset_index(drop=True)

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
            print(
                "training label shape: {}\nvalid label shape: {}\ntest label shape: {}\n".format(self.train_label.shape,
                                                                                                 self.valid_label.shape,
                                                                                                 self.test_label.shape))

        self.tickers = ordered_tickers

    def create_batch(self, size, data):
        # create new training set
        moving_window = []
        for i in range(size):
            one_step_moving_window = []
            for one_stock in data:
                one_step_moving_window.append(np.array(one_stock[i:i + self.lookback]))
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

    # Utils
    def minmax_scaler(self, df):
        min_ = df.min()
        max_ = df.max()
        return (df - min_) / (max_ - min_), min_.values.tolist(), max_.values.tolist()

    def normalize(self, df):
        mean_ = df.mean()
        std_ = df.std()
        return (df - mean_) / std_, mean_.values.tolist(), std_.values.tolist()

    def sample_neighbors(self, to_tensor):
        num_samples = self.config.num_sample_neighbors
        neighbors_batch = []
        for rel_neigh in self.neighbors:
            rel_neigh_batch = []
            for neigh in rel_neigh:
                short = max(num_samples - neigh.shape[0], 0)
                if short:
                    neighbors = np.concatenate([neigh, np.zeros(short)])
                    rel_neigh_batch.append(neighbors)
                else:
                    neighbors = np.random.choice(neigh, num_samples)
                    rel_neigh_batch.append(neighbors)
            rel_neigh_batch = np.stack(rel_neigh_batch, axis=0)
            neighbors_batch.append(rel_neigh_batch)
        neighbors_batch = np.stack(neighbors_batch, axis=0) #.astype(np.int32)
        if to_tensor:
            neighbors_batch = torch.from_numpy(neighbors_batch).type(torch.LongTensor)
        return neighbors_batch

    @staticmethod
    def classify(x):
        if x < 0:
            return 0
        elif x == 0:
            return 1
        else:
            return 2

    # Return values
    def get_dataset(self):
        return (self.train_set, self.train_label), (self.test_set, self.test_label), \
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
