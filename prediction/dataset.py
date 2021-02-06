import pandas as pd
import os


class DataSet:
    def __init__(self, config):
        self.config = config
        self.dir = self.config.market_directory

    def load(self):
        data = pd.read_csv(os.path.join(self.dir, 'ADVANC.BK.csv'))[self.config.feature_list]
        data = data.dropna()

        return data
