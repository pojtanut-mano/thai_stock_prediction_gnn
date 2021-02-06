from dataset import DataSet
import config


if __name__ == '__main__':
    stock = DataSet(config)
    df = stock.load()
    print(df)