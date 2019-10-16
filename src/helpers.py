import numpy as np
from src.params import *


def separate_data(tX_shuffled, y_shuffled, ratio=RATIO):
    length = len(y_shuffled)
    train_set_index = (int)(length * ratio)
    train_set = tX_shuffled[:train_set_index]
    train_label = y_shuffled[:train_set_index]
    test_set = tX_shuffled[train_set_index:]
    test_label =y_shuffled[train_set_index:] 
    return train_set, train_label, test_set, test_label


def test_accuracy(d, y_test, test_label):
    accuracy = np.abs(y_test + test_label)
    unique, counts = np.unique(accuracy, return_counts=True)
    print("Accuracy for degree {} : {}".format(d, counts[1] / len(accuracy)))


def flatten_list(list_to_flatten):
    flat_list = []
    for sublist in list_to_flatten:
        for item in sublist:
            flat_list.append(item)
