# -*- coding: utf-8 -*-
"""some helper functions for project 1."""

import csv
import numpy as np


def flatten_list(list_to_flatten):
    flat_list = []
    for sublist in list_to_flatten:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def remove_duplicate_columns(tX):
    features = tX.T
    new_features, indices = np.unique(features, return_inverse=True, axis=0)
    return new_features.T


def remove_duplicate_columns_grouped(tX_grouped):
    new_tX = []
    for i in range(len(tX_grouped)):
        new_tX.append(remove_duplicate_columns(tX_grouped[i]))
    return new_tX


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    print('\tLoading data...')

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

    print('\tData loaded.')
    return yb, input_data, ids


def predict_labels(weights, data, implementation):
    """Generates class predictions given weights, and a test data matrix
    Returns the actual computed values of y, and the classification version"""
    if implementation == 2:
        temp = np.dot(data, weights)
        y_pred = np.exp(temp) / (1 + np.exp(temp))
        y_pred = y_pred * 2 - 1
    else:
        y_pred = np.dot(data, weights)
    y_pred_clipped = np.copy(y_pred)
    y_pred_clipped[np.where(y_pred_clipped <= 0)] = -1
    y_pred_clipped[np.where(y_pred_clipped > 0)] = 1
    
    return y_pred, y_pred_clipped


def predict_labels_grouped(optimal_ws, tX_sub_test_grouped, implementation):
    y_pred_grouped = []
    y_pred_clipped_grouped = []
    for i in range(len(optimal_ws)):
        y_pred, y_pred_clipped = predict_labels(optimal_ws[i], tX_sub_test_grouped[i], implementation)
        y_pred_grouped.append(y_pred)
        y_pred_clipped_grouped.append(y_pred_clipped)
    return y_pred_grouped, y_pred_clipped_grouped


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
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
    print('\tCSV submission created.')
