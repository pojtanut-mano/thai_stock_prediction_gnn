import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import os

from dataset import StockDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd


class Evaluator:
    def __init__(self, config):
        self.config = config

    def metrics(self, pred, label):
        accuracy = accuracy_score(label, pred)
        f1 = f1_score(label, pred, labels=[0], average='micro')
        return accuracy, f1

    def export_metrics(self, data, model, result_df):
        (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = data
        train_df, valid_df, test_df = result_df

        train_dataset = StockDataset(X_train, y_train)
        valid_dataset = StockDataset(X_valid, y_valid)
        test_dataset = StockDataset(X_test, y_test)

        train_data_loader = DataLoader(train_dataset)
        valid_data_loader = DataLoader(valid_dataset)
        test_data_loader = DataLoader(test_dataset)

        # Pass train set through model
        train_pred_df, train_label_df = self.model_passing(train_data_loader, train_df, model)
        valid_pred_df, valid_label_df = self.model_passing(valid_data_loader, valid_df, model)
        test_pred_df, test_label_df = self.model_passing(test_data_loader, test_df, model)

        # Export raw
        raw_output_filename = os.path.join(self.config.checkpoint_dir, self.config.directory, self.config.raw_output_dir)
        os.mkdir(raw_output_filename)

        train_pred_df.to_csv(os.path.join(raw_output_filename, 'train_pred_df.csv'))
        valid_pred_df.to_csv(os.path.join(raw_output_filename, 'valid_pred_df.csv'))
        test_pred_df.to_csv(os.path.join(raw_output_filename, 'test_pred_df.csv'))

        train_label_df.to_csv(os.path.join(raw_output_filename, 'train_label_df.csv'))
        valid_label_df.to_csv(os.path.join(raw_output_filename, 'valid_label_df.csv'))
        test_label_df.to_csv(os.path.join(raw_output_filename, 'test_label_df.csv'))

        # Export classification report
        flat_train_pred, flat_train_label = train_pred_df.values.flatten().astype(int), train_label_df.values.flatten().astype(int)
        flat_valid_pred, flat_valid_label = valid_pred_df.values.flatten().astype(int), valid_label_df.values.flatten().astype(int)
        flat_test_pred, flat_test_label = test_pred_df.values.flatten().astype(int), test_label_df.values.flatten().astype(int)

        train_report = pd.DataFrame(classification_report(flat_train_label, flat_train_pred, output_dict=True)).transpose()
        valid_report = pd.DataFrame(classification_report(flat_valid_label, flat_valid_pred, output_dict=True)).transpose()
        test_report = pd.DataFrame(classification_report(flat_test_label, flat_test_pred, output_dict=True)).transpose()

        report_filename = os.path.join(self.config.checkpoint_dir, self.config.directory, self.config.report_dir)
        os.mkdir(report_filename)

        train_report.to_csv(os.path.join(report_filename, 'train_report.csv'))
        valid_report.to_csv(os.path.join(report_filename, 'valid_report.csv'))
        test_report.to_csv(os.path.join(report_filename, 'test_report.csv'))

        # Export Confusion Matrix
        train_conf = pd.DataFrame(confusion_matrix(flat_train_label, flat_train_pred)).transpose()
        valid_conf = pd.DataFrame(confusion_matrix(flat_valid_label, flat_valid_pred)).transpose()
        test_conf = pd.DataFrame(confusion_matrix(flat_test_label, flat_test_pred)).transpose()

        conf_filename = os.path.join(self.config.checkpoint_dir, self.config.directory, self.config.confusion_mat_dir)
        os.mkdir(conf_filename)

        train_conf.to_csv(os.path.join(conf_filename, 'train_conf.csv'))
        valid_conf.to_csv(os.path.join(conf_filename, 'valid_conf.csv'))
        test_conf.to_csv(os.path.join(conf_filename, 'test_conf.csv'))

    def model_passing(self, data_loader, result_df, model):
        label_df = result_df.copy()
        model.eval()
        counter = 0
        for input_hist, label in data_loader:
            input_hist = torch.squeeze(input_hist, dim=0).type(torch.float32).to(model.device)

            y_pred_proba = model.forward(input_hist).detach().cpu().numpy()
            y_pred = np.argmax(y_pred_proba, axis=1)

            result_df.iloc[counter, :] = y_pred
            label_df.iloc[counter, :] = label

            counter += 1

        return result_df, label_df

    def export_history(self, hist):
        cost_hist, train_acc_hist, valid_acc_hist, train_f1_hist, valid_f1_hist = hist
        self.hist_filename = os.path.join(self.config.checkpoint_dir, self.config.directory, self.config.hist_dir)

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
