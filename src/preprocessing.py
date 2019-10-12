import numpy as np


def data_separation(y, tX, ids, unwanted_value):
    """
    Separates the dataset into groups according to the appearance of the unwanted value in each data point.

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
        The labels grouped
    tX_grouped: object (array of arrays of data points)
        The features matrix grouped
    ids_grouped: 2D-array
        The data points' ids grouped
    masks: 2D-matrix
        The features of each group expressed in terms of 1 (means there is a UNWANTED_VALUE at this position) and 0
    counts: array
        The number of data points belonging to each group
    """

    unwanted_value_check = 1 * (tX == unwanted_value)
    masks, indices, counts = np.unique(unwanted_value_check, return_inverse=True, return_counts=True, axis=0)

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

    return np.asarray(y_grouped), np.array(tX_grouped, dtype=object), np.asarray(ids_grouped), masks, counts


def drop_features(tX_grouped, masks):
    """
    In each group of data points, drop the features/columns that contain only UNWANTED_VALUEs.

    Parameters
    ----------
    tX_grouped: array of arrays of data points
        The features matrix grouped
    masks: 2D-matrix
        The features of each group expressed in terms of 1 (means there is a UNWANTED_VALUE at this position) and 0

    Returns
    -------
    tX_grouped_clean: array of arrays of data points
       The features matrix grouped and cleaned of every UNWANTED_VALUE
    """

    tX_grouped_clean = []
    for i in range(len(tX_grouped)):
        temp = tX_grouped[i].T
        temp = np.delete(temp, np.where(masks[i] == 1), 0)
        tX_grouped_clean.append(temp.T)
    return tX_grouped_clean


def remove_invariable_features(tX_grouped):
    """
    In each group of data points, drop the features/columns that never change.

    Parameters
    ----------
    tX_grouped: array of arrays of data points
        The features matrix grouped

    Returns
    -------
    tX_clean: array of arrays of data points
       The features matrix grouped and cleaned of every invariable features
    """

    tX_clean = []
    for i in range(len(tX_grouped)):
        features = tX_grouped[i].T
        stds = np.std(features, axis=1)
        indices = np.where(stds == 0)
        features = np.delete(features, indices, 0)
        tX_clean.append(features.T)
    return tX_clean


def standardize(tX_grouped):
    """
    In each group of data points, standardize each feature/column.

    Parameters
    ----------
    tX_grouped: array of arrays of data points
        The features matrix grouped

    Returns
    -------
    tX_clean_std: array of arrays of data points
       The features matrix grouped standardized
    """

    tX_clean_std = []
    for i in range(len(tX_grouped)):
        temp = tX_grouped[i].T
        temp_len = len(temp)
        means = np.reshape(np.mean(temp, axis=1), [temp_len, 1])
        stds = np.reshape(np.std(temp, axis=1), [temp_len, 1])
        temp_standardized = (temp-means)/stds
        tX_clean_std.append(temp_standardized.T)
    return tX_clean_std
