import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import StockDataset

import os
from torchviz import make_dot
import time
import datetime


class Trainer:
    def __init__(self, model, data, config, evaluator):
        self.model = model
        self.data = data
        self.config = config
        self.evaluator = evaluator

        self.cost_hist = []
        self.valid_cost_hist = []
        self.train_acc_hist = []
        self.valid_acc_hist = []

        self.train_f1_hist = []
        self.valid_f1_hist = []

        (self.X_train, self.y_train), (self.X_test, self.y_test), (self.X_valid, self.y_valid) = data.get_dataset()
        self.dataset = StockDataset(self.X_train, self.y_train)
        self.data_loader = DataLoader(self.dataset, shuffle=config.shuffle_batch)
        self.valid_loader = DataLoader(StockDataset(self.X_valid, self.y_valid))

    def train_classifier(self):
        print('Start training classifier...\n')

        best_loss = 1000000
        best_valid_acc = 0
        stopping_criteria = 0

        start_time = time.time()
        for epoch in range(self.config.epochs):
            loss = self.train_epoch()

            self.model.eval()
            train_pred_list = []
            train_label_list = []

            valid_pred_list = []
            valid_label_list = []
            for input_hist, label in self.data_loader:
                input_hist = torch.squeeze(input_hist, dim=0).type(torch.float32).to(self.model.device)
                label = torch.squeeze(label, dim=0).type(torch.LongTensor).to(self.model.device)
                train_probs = self.model.forward(input_hist)
                train_preds = torch.argmax(train_probs, dim=1)

                train_pred_list.extend(train_preds.cpu().detach().tolist())
                train_label_list.extend(label.cpu().detach().tolist())

            for input_hist, label in self.valid_loader:
                input_hist = torch.squeeze(input_hist, dim=0).type(torch.float32).to(self.model.device)
                label = torch.squeeze(label, dim=0).type(torch.LongTensor).to(self.model.device)
                valid_probs = self.model.forward(input_hist)
                valid_preds = torch.argmax(valid_probs, dim=1)

                valid_pred_list.extend(valid_preds.cpu().detach().tolist())
                valid_label_list.extend(label.cpu().detach().tolist())

            train_acc, train_f1 = self.evaluator.metrics(train_pred_list, train_label_list)
            valid_acc, valid_f1 = self.evaluator.metrics(valid_pred_list, valid_label_list)

            self.cost_hist.append(loss)
            self.train_acc_hist.append(train_acc)
            self.valid_acc_hist.append(valid_acc)
            self.train_f1_hist.append(train_f1)
            self.valid_f1_hist.append(valid_f1)

            # Scheduler step
            self.model.scheduler.step()

            # Early stopping
            if best_loss - loss < self.config.early_stopping_threshold:
                stopping_criteria += 1
                print('Model not improving for {} epoch(s).'.format(stopping_criteria))
            else:
                best_loss = loss
                stopping_criteria = 0

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                self.model.save_checkpoint()

            if stopping_criteria >= self.config.early_stopping_period:
                print('Early stopping occurs at {}'.format(epoch + 1))
                print('Last epoch {:3d}, loss = {:.8f}, accuracy = {:.2f}, validation = {:.2f}'.format(epoch+1, self.cost_hist[-1], train_acc * 100, valid_acc * 100))
                break

            if (epoch+1) % self.config.print_log == 0:
                sample_index = np.random.choice(range(label.shape[0]))
                print('\nEpoch {:3d}, loss = {:.8f}, accuracy = {:.2f}, validation = {:.2f}'.format(epoch+1, self.cost_hist[-1], train_acc * 100, valid_acc * 100))
                print('Example {}: probs = {}, label = {}'.format(sample_index, valid_probs[sample_index].cpu().detach(), label[sample_index]))
                print(np.unique(train_pred_list, return_counts=True))
                print('Current learning rate: {}'.format(self.model.scheduler.get_last_lr()[0]))
                print('---------------------------------------')

            if np.abs(train_acc - valid_acc) > self.config.overfitting_threshold:
                print('Overfitting occurs at {} epoch'.format(epoch + 1))
                break

            if (epoch+1) == self.config.epochs:
                print('Max iterations reached.')

        finish_time = time.time()
        second_used = finish_time - start_time

        print('\nTime used: {}'.format(datetime.timedelta(seconds=second_used)))

    def train_regressor(self):
        best_loss = 1000000
        stopping_criteria = 0
        start_time = time.time()

        for epoch in range(self.config.epochs):
            loss = self.train_epoch()
            self.cost_hist.append(loss)

            self.model.eval()
            valid_loss_hist = []
            for input_hist, label in self.valid_loader:
                input_hist = torch.squeeze(input_hist, dim=0).type(torch.float32).to(self.model.device)
                label = torch.squeeze(label, dim=0).type(torch.float32).to(self.model.device)
                valid_preds = self.model.forward(input_hist)

                valid_loss = self.model.loss(valid_preds, label)
                if self.config.loss_function == 'RankMSE':
                    valid_loss = valid_loss + self.config.rank_weight * self.rank_loss(valid_preds, label)

                valid_loss_hist.append(valid_loss.cpu().detach())

            valid_loss = np.mean(valid_loss_hist)
            self.valid_cost_hist.append(valid_loss)

            # Scheduler step
            self.model.scheduler.step()

            # Early stopping
            if best_loss - loss < self.config.early_stopping_threshold:
                stopping_criteria += 1
                print('Model not improving for {} epoch(s).'.format(stopping_criteria))
            else:
                best_loss = loss
                stopping_criteria = 0
                self.model.save_checkpoint()

            if stopping_criteria >= self.config.early_stopping_period:
                print('Early stopping occurs')
                print('Last epoch {:3d}, loss = {:.8f}, valid loss = {:.8f}'.format(epoch + 1, self.cost_hist[-1], self.valid_cost_hist[-1]))
                break

            if (epoch + 1) % self.config.print_log == 0:
                sample_index = np.random.choice(range(label.shape[0]))
                print('\nEpoch {:3d}, loss = {:.8f}'.format(epoch + 1, self.cost_hist[-1],))
                print('Example {}: prediction = {}, label = {}'.format(sample_index, valid_preds[sample_index].cpu().detach(), label[sample_index]))
                print('Current learning rate: {}'.format(self.model.scheduler.get_last_lr()[0]))
                print('---------------------------------------')

            if (epoch + 1) == self.config.epochs:
                print('Max iterations reached.')

        finish_time = time.time()
        second_used = finish_time - start_time

        print('\nTime used: {}'.format(datetime.timedelta(seconds=second_used)))

    def train_epoch(self):
        tmp_cost_hist = []
        for input_hist, label in self.data_loader:
            # Squeeze one dim
            input_hist = torch.squeeze(input_hist, dim=0).type(torch.float32).to(self.model.device)
            label = torch.squeeze(label, dim=0).type(torch.LongTensor if self.config.target_type == 'classification' else torch.float32).to(self.model.device)

            input_hist.requires_grad_(True)
            # Train mode
            self.model.train()
            pred = self.model.forward(input_hist)
            cost = self.model.loss(pred, label)
            if self.config.loss_function == 'RankMSE' and self.config.target_type == 'regression':
                # Rank loss
                rank_loss = self.rank_loss(pred, label)
                cost = cost + self.config.rank_weight * rank_loss

            # grad = []
            # for p in self.model.parameters():
            #     print(p.grad)

            self.model.optimizer.zero_grad()
            cost.backward()
            self.model.optimizer.step()

            tmp_cost_hist.append(cost.cpu().detach())
        return np.mean(tmp_cost_hist)

    def rank_loss(self, pred, label):
        ones_vector = torch.ones(pred.shape[0], 1).to(self.model.device)

        pred_rank_diff = torch.subtract(torch.matmul(pred, torch.transpose(ones_vector, 0, 1)),
                                        torch.matmul(ones_vector, torch.transpose(pred, 0, 1)))

        label_rank_diff = torch.subtract(torch.matmul(label, torch.transpose(ones_vector, 0, 1)),
                                         torch.matmul(ones_vector, torch.transpose(label, 0, 1)))

        rank_diff = -1 * torch.mul(pred_rank_diff, label_rank_diff)
        relu = torch.nn.ReLU()
        rank_diff = relu(rank_diff)
        rank_loss = torch.mean(rank_diff)
        return rank_loss
    
    def get_hist(self):
        return self.cost_hist, self.train_acc_hist, self.valid_acc_hist, self.train_f1_hist, self.valid_f1_hist
