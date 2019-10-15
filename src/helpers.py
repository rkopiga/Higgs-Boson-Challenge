import numpy as np
from src.params import *

def shuffle_data(y, tX, ids):
    y = y.reshape(y.shape[0],1)
    ids = ids.reshape(ids.shape[0],1)
    model_data = np.hstack((tX, y, ids))
    np.random.shuffle(model_data)
    ids_shuffled = model_data[:,model_data.shape[1]-1]
    y_shuffled = model_data[:,model_data.shape[1]-2]
    tX_shuffled = model_data[:,:model_data.shape[1]-2]
    return tX_shuffled, y_shuffled, ids_shuffled

def separate_data(tX_shuffled, y_shuffled, ratio=RATIO):
    length = len(y_shuffled)
    train_set_index = (int)(length * ratio)
    train_set = tX_shuffled[:train_set_index]
    train_label = y_shuffled[:train_set_index]
    test_set = tX_shuffled[train_set_index:]
    test_label =y_shuffled[train_set_index:] 
    return train_set, train_label, test_set, test_label

def test_accuracy(y_test, test_label):
    accuracy = np.abs(y_test + test_label)
    unique, counts = np.unique(accuracy, return_counts=True)
    print("Group: ", i, ". Accuracy for degree " , d , " : ", counts[1]/len(accuracy))