import numpy as np
from sklearn.metrics import accuracy_score
from torchviz import make_dot

class Evaluator:
    def __init__(self, config):
        self.config = config

    def metrics(self, pred, label):
        accuracy = accuracy_score(label, pred)
        return accuracy
