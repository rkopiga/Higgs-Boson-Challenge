# -*- coding: utf-8 -*-
"""
Some helper functions for project 1.
"""

import csv
import numpy as np


def flatten_list(list_to_flatten):
    """
    Flatten the given list.

    Parameters
    ----------
    list_to_flatten: list
        The list to flatten

    Returns
    -------
    flat_list: list
        The flattened list
    """
    flat_list = []
    for sublist in list_to_flatten:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def remove_duplicate_columns(tX):
    """
    Remove the duplicate features, i.e the columns having the same values in tX.

    Parameters
    ----------
    tX: array
        The features matrix

    Returns
    -------
    new_tX: array

    """
    features = tX.T
    new_features, indices = np.unique(features, return_inverse=True, axis=0)
    new_tX = new_features.T
    return new_tX


def remove_duplicate_columns_grouped(tX_grouped):
    """
    Remove the duplicate features, i.e the columns having the same values in tX, in each group.

    Parameters
    ----------
    tX_grouped: list
        The list of features matrices

    Returns
    -------
    new_tX_grouped: list
        The new list of features matrices
    """
    new_tX = []
    for i in range(len(tX_grouped)):
        new_tX.append(remove_duplicate_columns(tX_grouped[i]))
    return new_tX


def load_csv_data(data_path, sub_sample=False):
    """
    Load data and returns y (class labels), tX (features) and ids (event ids)

    Parameters
    ----------
    data_path: str
        The path to the dataset
    sub_sample: boolean
        Rather we only want a small chunk of data from the dataset (50 data points)

    Returns
    -------
    yb: array
        The labels
    input_data: array
        The features matrix
    ids: array
        The ids of the data points
    """
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
    """
    Generate class predictions given weights, and a test data matrix.
    Return the actual computed values of y, and the classification version

    Parameters
    ----------
    weights: array
        The previously computed weights
    data: array
        The data on which we want to predict the labels
    implementation: int
        The used implementation (0, 1, 2, ...)

    Returns
    -------
    y_pred: array
        The labels before clipping to 1 and -1
    y_pred_clipped: array
        The labels clipped to 1 and -1
    """
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
    """
    Generate class predictions given weights, and a test data matrix, for each group.

    Parameters
    ----------
    optimal_ws: list
        The previously computed weights for each group
    tX_sub_test_grouped: list
        The list of datas on which we want to predict the labels
    implementation: int
        The used implementation (0, 1, 2, ...)

    Returns
    -------
    y_pred_grouped: list
        The labels before clipping to 1 and -1 for each group
    y_pred_clipped_grouped: list
        The labels clipped to 1 and -1 for each group
    """
    y_pred_grouped = []
    y_pred_clipped_grouped = []
    for i in range(len(optimal_ws)):
        y_pred, y_pred_clipped = predict_labels(optimal_ws[i], tX_sub_test_grouped[i], implementation)
        y_pred_grouped.append(y_pred)
        y_pred_clipped_grouped.append(y_pred_clipped)
    return y_pred_grouped, y_pred_clipped_grouped


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to Aicrowd

    Parameters
    ----------
    ids: array
        Event ids associated with each prediction
    y_pred: array
        Predicted class labels
    name: str
        The name of the csv file

    Returns
    -------
    None
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
    print('\tCSV submission created.')
