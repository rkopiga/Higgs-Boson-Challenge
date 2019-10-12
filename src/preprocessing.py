import numpy as np


def data_separation(y, tX, ids, unwanted_value):
    """
    Separates the dataset into groups according to the appearance
    of the unwanted value in each data point.

    Parameters
    ----------
    y: array
        The labels
    tX: 2D-matrix
        The features matrix
    ids: array
        The ids of the data points

    Returns
    -------
    y_grouped: 2D-array
    tX_grouped: object (array of arrays of data points)
    ids_grouped: 2D-array
    unique: 2D-matrix
    """

    unwanted_value_check = 1 * (tX == unwanted_value)
    unique, indices, counts = np.unique(unwanted_value_check, return_inverse=True, return_counts=True, axis=0)

    max_index = max(indices)
    y_grouped = []
    tX_grouped = []
    ids_grouped = []
    n_data_points = len(tX)
    for i in range(max_index + 1):
        condition = (indices == i)
        y_grouped.append(np.extract(condition, y))
        ids_grouped.append(np.extract(condition, ids))

        indices_to_take = np.extract(condition, range(n_data_points))
        tX_grouped.append(np.take(tX, indices_to_take, axis=0))

    return np.asarray(y_grouped), np.array(tX_grouped, dtype=object), np.asarray(ids_grouped), unique, counts


def drop_features(tX_grouped, masks):
    tX_grouped_clean = []
    for i in range(len(tX_grouped)):
        temp = tX_grouped[i].T
        temp = np.delete(temp, np.where(masks[i] == 1), 0)
        tX_grouped_clean.append(temp.T)
    return tX_grouped_clean


def remove_invariable_features(tX_grouped_clean):
    tX_clean = []
    for i in range(len(tX_grouped_clean)):
        temp = tX_grouped_clean[i].T
        stds = np.std(temp, axis=1)
        indices = np.where(stds==0)
        temp = np.delete(temp, indices, 0)
        tX_clean.append(temp.T)
    return tX_clean


def standardize(tX_clean):
    tX_clean_std = []
    for i in range(len(tX_clean)):
        temp = tX_clean[i].T
        temp_len = len(temp)
        means = np.reshape(np.mean(temp, axis=1), [temp_len, 1])
        stds = np.reshape(np.std(temp, axis=1), [temp_len, 1])
        temp_standardized = (temp-means)/stds
        tX_clean_std.append(temp_standardized.T)
    return tX_clean_std
