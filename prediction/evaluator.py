import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import os
import pickle

from dataset import StockDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd


class Evaluator:
    def __init__(self, config):
        self.config = config

    def metrics(self, pred, label):
        accuracy = accuracy_score(label, pred)
        f1 = []
        for i in range(3):
            f1.append(f1_score(label, pred, labels=[i], average='micro'))
        return accuracy, np.mean(f1)

    def export_metrics(self, data, model, result_df, period):
        (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = data
        train_df, valid_df, test_df = result_df

        train_dataset = StockDataset(X_train, y_train)
        valid_dataset = StockDataset(X_valid, y_valid)
        test_dataset = StockDataset(X_test, y_test)

        train_data_loader = DataLoader(train_dataset)
        valid_data_loader = DataLoader(valid_dataset)
        test_data_loader = DataLoader(test_dataset)

        # Pass train set through model
        if self.config.target_type == 'classification':
            train_pred_df, train_label_df, train_pred_proba_df = self.model_passing(train_data_loader, train_df, model)
            valid_pred_df, valid_label_df, valid_pred_proba_df = self.model_passing(valid_data_loader, valid_df, model)
            test_pred_df, test_label_df, test_pred_proba_df = self.model_passing(test_data_loader, test_df, model)
        else:
            train_pred_df, train_label_df = self.model_passing(train_data_loader, train_df, model)
            valid_pred_df, valid_label_df= self.model_passing(valid_data_loader, valid_df, model)
            test_pred_df, test_label_df = self.model_passing(test_data_loader, test_df, model)

        # Export raw
        raw_output_filename = os.path.join(self.config.checkpoint_dir, self.config.directory, period, self.config.raw_output_dir)
        os.mkdir(raw_output_filename)

        train_pred_df.to_csv(os.path.join(raw_output_filename, 'train_pred_df.csv'))
        valid_pred_df.to_csv(os.path.join(raw_output_filename, 'valid_pred_df.csv'))
        test_pred_df.to_csv(os.path.join(raw_output_filename, 'test_pred_df.csv'))

        train_label_df.to_csv(os.path.join(raw_output_filename, 'train_label_df.csv'))
        valid_label_df.to_csv(os.path.join(raw_output_filename, 'valid_label_df.csv'))
        test_label_df.to_csv(os.path.join(raw_output_filename, 'test_label_df.csv'))

        if self.config.target_type == 'classification':
            train_pred_proba_df.to_csv(os.path.join(raw_output_filename, 'train_pred_proba_df.csv'))
            valid_pred_proba_df.to_csv(os.path.join(raw_output_filename, 'valid_pred_proba_df.csv'))
            test_pred_proba_df.to_csv(os.path.join(raw_output_filename, 'test_pred_proba_df.csv'))

            # Export classification report
            flat_train_pred, flat_train_label = train_pred_df.values.flatten().astype(int), train_label_df.values.flatten().astype(int)
            flat_valid_pred, flat_valid_label = valid_pred_df.values.flatten().astype(int), valid_label_df.values.flatten().astype(int)
            flat_test_pred, flat_test_label = test_pred_df.values.flatten().astype(int), test_label_df.values.flatten().astype(int)

            train_report = pd.DataFrame(classification_report(flat_train_label, flat_train_pred, output_dict=True)).transpose()
            valid_report = pd.DataFrame(classification_report(flat_valid_label, flat_valid_pred, output_dict=True)).transpose()
            test_report = pd.DataFrame(classification_report(flat_test_label, flat_test_pred, output_dict=True)).transpose()

            report_filename = os.path.join(self.config.checkpoint_dir, self.config.directory, period, self.config.report_dir)
            os.mkdir(report_filename)

            train_report.to_csv(os.path.join(report_filename, 'train_report.csv'))
            valid_report.to_csv(os.path.join(report_filename, 'valid_report.csv'))
            test_report.to_csv(os.path.join(report_filename, 'test_report.csv'))

            # Export Confusion Matrix
            train_conf = pd.DataFrame(confusion_matrix(flat_train_label, flat_train_pred)).transpose()
            valid_conf = pd.DataFrame(confusion_matrix(flat_valid_label, flat_valid_pred)).transpose()
            test_conf = pd.DataFrame(confusion_matrix(flat_test_label, flat_test_pred)).transpose()

            conf_filename = os.path.join(self.config.checkpoint_dir, self.config.directory, period, self.config.confusion_mat_dir)
            os.mkdir(conf_filename)

            train_conf.to_csv(os.path.join(conf_filename, 'train_conf.csv'))
            valid_conf.to_csv(os.path.join(conf_filename, 'valid_conf.csv'))
            test_conf.to_csv(os.path.join(conf_filename, 'test_conf.csv'))

    def model_passing(self, data_loader, result_df, model):
        label_df = result_df.copy()
        proba_result_df = result_df.copy()
        model.eval()
        counter = 0
        for input_hist, label in data_loader:
            pred_proba = []
            input_hist = torch.squeeze(input_hist, dim=0).type(torch.float32).to(model.device)
            if self.config.target_type == 'classification':
                y_pred_proba = model.forward(input_hist).detach().cpu().numpy()
                for i in range(len(y_pred_proba)):
                    pred_proba.append('{}'.format(y_pred_proba[i, :]))
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.forward(input_hist).detach().cpu().numpy()
                y_pred = np.squeeze(y_pred, axis=1)
                label = torch.squeeze(torch.squeeze(label, dim=0), dim=1)

            result_df.iloc[counter, :] = y_pred
            label_df.iloc[counter, :] = label

            if self.config.target_type == 'classification':
                proba_result_df.iloc[counter, :] = pred_proba

                counter += 1
        if self.config.target_type == 'classification':
            return result_df, label_df, proba_result_df
        return result_df, label_df

    def simple_model_passing(self, data_loader, model):
        model.eval()
        result_list = []
        label_list = []
        for input_hist, label in data_loader:
            input_hist = torch.squeeze(input_hist, dim=0).type(torch.float32).to(model.device)
            if self.config.target_type == 'classification':
                y_pred_proba = model.forward(input_hist).detach().cpu().numpy()
                y_pred = np.argmax(y_pred_proba, axis=1)
                result_list.extend(y_pred)
                label_list.extend(label.numpy().ravel())
            else:
                y_pred = model.forward(input_hist).detach().cpu().numpy()
                result_list.extend(y_pred.ravel())
                label_list.extend(label.numpy().ravel())
        return result_list, label_list

    def export_history(self, hist, period):
        cost_hist, train_acc_hist, valid_acc_hist, train_f1_hist, valid_f1_hist = hist
        self.hist_filename = os.path.join(self.config.checkpoint_dir, self.config.directory, period, self.config.hist_dir)

        os.mkdir(self.hist_filename)
        # Plot cost hist
        self.cost_plot(cost_hist)

        if self.config.target_type == 'classification':
            # Plot accuracy over time
            self.comparison_plot(train_acc_hist, valid_acc_hist, 'accuracy')

            # Plot f1 over time
            self.comparison_plot(train_f1_hist, valid_f1_hist, 'f1')

    def cost_plot(self, df):
        fig, ax = plt.subplots(1, 1, figsize=self.config.fig_size)
        ax.plot(range(1, len(df)+1), df, label='Cost')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cost from training step')
        ax.set_title('Cost over epochs')
        if self.config.target_type == 'regression':
            ax.set_ylim((0, 0.003))
        ax.legend()
        plt.savefig(os.path.join(self.hist_filename, 'cost_plot.png'))
        plt.clf()

    def comparison_plot(self, train, valid, plot_name):
        fig, ax = plt.subplots(1, 1, figsize=self.config.fig_size)
        ax.plot(range(1, len(train) + 1), train, label='Train set', color='red')
        ax.plot(range(1, len(valid) + 1), valid, label='Validation set', color='blue')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('{}'.format(plot_name))
        ax.set_title('{} over epochs'.format(plot_name))
        ax.legend()
        plt.savefig(os.path.join(self.hist_filename, '{}_plot.png'.format(plot_name)))
        plt.clf()

    def save_result(self, df, model, data, param, index):
        (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = data

        train_dataset = StockDataset(X_train, y_train)
        valid_dataset = StockDataset(X_valid, y_valid)
        test_dataset = StockDataset(X_test, y_test)

        train_data_loader = DataLoader(train_dataset)
        valid_data_loader = DataLoader(valid_dataset)
        test_data_loader = DataLoader(test_dataset)

        # Pass train set through model
        if self.config.target_type == 'classification':
            train_pred_df, train_label_df = self.simple_model_passing(train_data_loader, model)
            valid_pred_df, valid_label_df = self.simple_model_passing(valid_data_loader, model)
            test_pred_df, test_label_df = self.simple_model_passing(test_data_loader, model)

            train_acc, valid_acc, test_acc = accuracy_score(train_label_df, train_pred_df), accuracy_score(
                valid_label_df, valid_pred_df), accuracy_score(test_label_df, test_pred_df)
            train_f1, valid_f1, test_f1 = f1_score(train_label_df, train_pred_df, average='macro'), f1_score(
                valid_label_df, valid_pred_df, average='macro'), f1_score(test_label_df, test_pred_df, average='macro')

            df.append(param + [train_acc, train_f1, valid_acc, valid_f1, test_acc, test_f1])

        else:
            train_pred_df, train_label_df = self.simple_model_passing(train_data_loader, model)
            valid_pred_df, valid_label_df = self.simple_model_passing(valid_data_loader, model)
            test_pred_df, test_label_df = self.simple_model_passing(test_data_loader, model)

            train_mse, valid_mse, test_mse = mean_squared_error(train_label_df, train_pred_df), mean_squared_error(valid_label_df, valid_pred_df), mean_squared_error(test_label_df, test_pred_df)
            df.append(param + [train_mse, valid_mse, test_mse])

        return df

    def save_config(self, param, period):
        with open(os.path.join(self.config.checkpoint_dir, self.config.directory, period, self.config.config_name), 'wb') as f:
            pickle.dump(param, f)

    def save_model(self, model, path):
        torch.save(model, path)