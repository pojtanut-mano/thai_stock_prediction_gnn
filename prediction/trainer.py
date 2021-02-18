import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import StockDataset
import os
from torchviz import make_dot


class Trainer:
    def __init__(self, model, data, config, evaluator):
        self.model = model
        self.data = data
        self.config = config
        self.evaluator = evaluator

        self.cost_hist = []
        self.train_acc_hist = []
        self.valid_acc_hist = []

        self.train_f1_hist = []
        self.valid_f1_hist = []

        (self.X_train, self.y_train), (self.X_test, self.y_test), (self.X_valid, self.y_valid) = data.get_dataset()
        self.dataset = StockDataset(self.X_train, self.y_train)
        self.data_loader = DataLoader(self.dataset, shuffle=config.shuffle_batch)
        self.valid_loader = DataLoader(StockDataset(self.X_valid, self.y_valid))

    def train(self):
        print('Start training...\n')
        # Create folder
        os.mkdir(os.path.join(self.config.checkpoint_dir, self.config.directory))

        best_loss = 1000000
        stopping_criteria = 0

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

            # Retrieve gradient
            # grad = []
            # for p in self.model.parameters():
            #     grad.append(float(p.grad.norm().cpu().detach()))
            # print('Epoch {:3d}, Avg. grad: {:.6f}'.format(epoch, sum(grad) / len(grad)))

            train_acc, train_f1 = self.evaluator.metrics(train_pred_list, train_label_list)
            valid_acc, valid_f1 = self.evaluator.metrics(valid_pred_list, valid_label_list)

            self.cost_hist.append(loss)
            self.train_acc_hist.append(train_acc)
            self.valid_acc_hist.append(valid_acc)
            self.train_f1_hist.append(train_f1)
            self.valid_f1_hist.append(valid_f1)

            # Early stopping
            if best_loss - loss >= self.config.early_stopping_threshold:
                print('\nEpoch {:3d}, loss = {:.8f}, accuracy = {:.2f}, validation = {:.2f}'.format(epoch + 1, np.mean(
                    self.cost_hist), train_acc * 100, valid_acc * 100))
                print(
                    'Example {}: probs = {}, label = {}'.format(sample_index, valid_probs[sample_index].cpu().detach(),
                                                                label[sample_index]))
                print(np.unique(train_pred_list, return_counts=True))
                print('---------------------------------------')
                stopping_criteria = 0
                best_loss = loss
                self.model.save_checkpoint()

            else:
                stopping_criteria += 1
                print('Model not improving for {} epoch(s).'.format(stopping_criteria))

            if stopping_criteria >= self.config.early_stopping_period:
                print('Early stopping occurs')
                print('Last epoch {:3d}, loss = {:.8f}, accuracy = {:.2f}, validation = {:.2f}'.format(epoch+1, np.mean(self.cost_hist), train_acc * 100, valid_acc * 100))
                break

            if (epoch+1) % self.config.print_log == 0:
                sample_index = np.random.choice(range(label.shape[0]))
                print('\nEpoch {:3d}, loss = {:.8f}, accuracy = {:.2f}, validation = {:.2f}'.format(epoch+1, np.mean(self.cost_hist), train_acc * 100, valid_acc * 100))
                print('Example {}: probs = {}, label = {}'.format(sample_index, valid_probs[sample_index].cpu().detach(), label[sample_index]))
                print(np.unique(train_pred_list, return_counts=True))
                print('---------------------------------------')

            if (epoch+1) == self.config.epochs:
                self.model.save_checkpoint()
                print('Max iterations reached.')

    def train_epoch(self):
        tmp_cost_hist = []
        for input_hist, label in self.data_loader:
            # Squeeze one dim
            input_hist = torch.squeeze(input_hist, dim=0).type(torch.float32).to(self.model.device)
            label = torch.squeeze(label, dim=0).type(torch.LongTensor).to(self.model.device)

            input_hist.requires_grad_(True)

            # Train mode
            self.model.train()
            pred = self.model.forward(input_hist)
            cost = self.model.loss(pred, label)

            self.model.optimizer.zero_grad()
            self.model.zero_grad()
            cost.backward()
            self.model.optimizer.step()

            tmp_cost_hist.append(cost.cpu().detach())
        return np.mean(tmp_cost_hist)

    def get_model(self):
        return self.model
    
    def get_hist(self):
        return self.cost_hist, self.train_acc_hist, self.valid_acc_hist, self.train_f1_hist, self.valid_f1_hist
