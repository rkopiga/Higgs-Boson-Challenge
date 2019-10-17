import numpy as np
from src.params import *


def separate_data(tX, y, ratio=RATIO):
    length = len(y)
    train_set_index = int(length * ratio)
    train_set = tX[:train_set_index]
    train_label = y[:train_set_index]
    test_set = tX[train_set_index:]
    test_label = y[train_set_index:]
    return train_set, train_label, test_set, test_label


def test_accuracy(d, y_test, test_label):
    accuracy = np.abs(y_test + test_label)
    unique, counts = np.unique(accuracy, return_counts=True)
    print("Accuracy for degree {} : {}".format(d, counts[1] / len(accuracy)))