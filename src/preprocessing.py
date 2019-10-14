import numpy as np
import src.params as params


def replace_unwanted_value_by_mean(tX, unwanted_value):
    features = tX.T
    mask = features == unwanted_value
    features_new = np.ma.array(features, mask=mask)
    means = np.mean(features_new, axis=1)
    for i in range(len(features)):
        a = features_new[i]
        a[a == unwanted_value] = means[i]
        features[i] = a
    return features.T


def replace_unwanted_value_by_mean_grouped(tX_grouped, unwanted_value):
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(replace_unwanted_value_by_mean(tX_grouped[i], unwanted_value))
    return tX_grouped_new


def data_separation_1(y, tX, ids, unwanted_value):
    """
    Separate the dataset into groups according to the appearance of the unwanted value in each data point.

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

    y_grouped = []
    tX_grouped = []
    ids_grouped = []
    for i in range(max(indices) + 1):
        condition = (indices == i)

        y_grouped.append(np.extract(condition, y))
        ids_grouped.append(np.extract(condition, ids))

        indices_to_take = np.extract(condition, range(len(tX)))
        tX_grouped.append(np.take(tX, indices_to_take, axis=0))

    return np.asarray(y_grouped), np.array(tX_grouped, dtype=object), np.asarray(ids_grouped), masks, counts


def data_separation_2(y, tX, ids):
    """
        Separate the dataset into groups according to the value of the feature PRI_jet_column (also called PRI_jet_num).

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
    masks = []
    counts = []
    y_grouped = []
    tX_grouped = []
    ids_grouped = []
    for i in range(params.PRI_jet_num_max_value + 1):
        condition = (tX.T[params.PRI_jet_num_index] == i)
        print(len(condition))
        masks.append(condition)

        counts.append(np.sum(condition))
        y_grouped.append(np.extract(condition, y))
        ids_grouped.append(np.extract(condition, ids))

        indices_to_take = np.extract(condition, range(len(tX)))
        tX_grouped.append(np.take(tX, indices_to_take, axis=0))

    return np.asarray(y_grouped), np.array(tX_grouped, dtype=object), np.asarray(ids_grouped), masks, counts


def remove_invariable_features(tX_grouped):
    """
    In each group of data points, drop the features/columns that never change (including the UNWANTED_VALUE)

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
