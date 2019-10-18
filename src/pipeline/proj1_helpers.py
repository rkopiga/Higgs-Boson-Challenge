# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
from src.params import *
from src.pipeline.implementations import *

import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, logistic_model=LOGISTIC_MODEL):
    """Generates class predictions given weights, and a test data matrix
    Returns the actual computed values of y, and the classification version"""
    if logistic_model:
        temp = np.dot(data, weights)
        y_pred = np.exp(temp) / (1 + np.exp(temp))
        y_pred = y_pred * 2 - 1
    else:
        y_pred = np.dot(data, weights)
    y_pred_clipped = np.copy(y_pred)
    y_pred_clipped[np.where(y_pred_clipped <= 0)] = -1
    y_pred_clipped[np.where(y_pred_clipped > 0)] = 1
    
    return y_pred, y_pred_clipped


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
