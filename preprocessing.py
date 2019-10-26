import numpy as np
import params

"""
Preprocessing pipeline
"""


def preprocess(
    y,
    tX,
    ids,
    shuffle=params.SHUFFLE_DATA,
    remove_phis=params.REMOVE_PHIS,
    add_diff_phis=params.ADD_DIFF_PHIS,
    unwanted_value=params.UNWANTED_VALUE,
    pri_jet_num_index=params.PRI_jet_num_index,
    group=params.GROUP,
    group_1=params.GROUP_1,
    group_2=params.GROUP_2,
    less_groups=params.LESS_GROUPS,
    replace_unwanted_value=params.REPLACE_UNWANTED_VALUE,
    value=params.VALUE,
    remove_inv_features=params.REMOVE_INV_FEATURES,
    std=params.STD,
    replace_outliers=params.REPLACE_OUTLIERS,
    outlier_value=params.OUTLIER_VALUE,
    threshold=params.THRESHOLD):
    """
    Preprocess the dataset

    Parameters
    ----------
    y: array
        The labels
    tX: 2D-matrix
        The features matrix
    ids: array
        The ids of the data points
    group_1: boolean
        Rather we split the dataset with the split function 1 or not
    group_2: boolean
        Rather we split the dataset with the split function 2 or not
    replace_unwanted_value: boolean
        Rather we replace all the unwanted values by the mean of the remaining values in each feature or not
    remove_inv_features: boolean
        Rather we remove all the invariable features or not
    std: boolean
        Rather we standardize each feature in each group or not
    replace_outliers: boolean
        Rather we replace the outliers in each group or not
    threshold: integer
        Parameter that defines an outlier

    Returns
    -------
    y: 2D-array
        The labels
    tX: 2D-list
        The features matrix
    ids: 2D-array
        The data points' ids
    masks: 2D-matrix
        The features of each group expressed in terms of 1 (means there is a UNWANTED_VALUE at this position) and 0
        or None depending on the chosen split function
    counts: array
        The number of data points belonging to each group or None depending on the chosen split function
    """
    print('\tPreprocessing...')
    masks = None
    counts = None

    if shuffle:
        y, tX, ids = shuffle_data(y, tX, ids)

    if remove_phis:
        initial_feature_number = 30
        tX = handle_angle_phis(tX, add_diff_phis)
        pri_jet_num_index = params.PRI_jet_num_new_index
        phi_replace_unwanted_value = replace_unwanted_value_by_value(tX[:,initial_feature_number+1:],unwanted_value,'median')
        tX[:,initial_feature_number+1:] = phi_replace_unwanted_value
    
    if group:
        if group_1:
            y, tX, ids, masks, counts = split_in_groups_1(y, tX, ids, unwanted_value)
        elif group_2:
            y, tX, ids, masks, counts = split_in_groups_2(y, tX, ids, pri_jet_num_index, less_groups)
            # check_uniqueness_in_group(tX, unwanted_value)
            if params.GROUP_2_ADDITIONAL_SPLITTING:
                y, tX, ids, masks, counts = additional_splitting(y, tX, ids, unwanted_value)
        if remove_inv_features:
            tX = remove_invariable_features_grouped(tX)
        if replace_unwanted_value:
            tX = replace_unwanted_value_by_value_grouped(tX, unwanted_value, value)
        if std:
            tX = standardize_grouped(tX)
        if replace_outliers: 
            tX = replace_outliers_grouped(tX, threshold, outlier_value)
    else:
        if remove_inv_features:
            tX = remove_invariable_features(tX)
        if replace_unwanted_value:
            tX = replace_unwanted_value_by_value(tX, unwanted_value, value)
        if std:
            tX = standardize(tX)
        if replace_outliers:
            tX = replace_outliers_by_threshold(tX, threshold, outlier_value)
    print('\tPreprocessing ok.')
    return y, tX, ids, masks, counts


def check_uniqueness_in_group(tX_grouped, unwanted_value):
    masks_check = []
    counts_check = []
    for i in range(len(tX_grouped)):
        unwanted_value_check = 1 * (tX_grouped[i] == unwanted_value)
        masks_and_counts = np.unique(unwanted_value_check, return_counts=True, axis=0)
        masks_check.append(masks_and_counts[0])
        counts_check.append(masks_and_counts[1])
    print(masks_check)
    print(counts_check)


def shuffle_data(y, tX, ids):
    y = y.reshape(y.shape[0], 1)
    ids = ids.reshape(ids.shape[0], 1)
    model_data = np.hstack((tX, y, ids))
    np.random.shuffle(model_data)
    ids_shuffled = model_data[:, model_data.shape[1]-1]
    y_shuffled = model_data[:, model_data.shape[1]-2]
    tX_shuffled = model_data[:, :model_data.shape[1]-2]
    return y_shuffled, tX_shuffled, ids_shuffled


def handle_angle_phis(tX, add_diff_phis):
    features = tX
    PHIS_features = np.take(tX, params.PHIs_indices, axis=1)
    diff_features = PHIS_features[:, 0].reshape(PHIS_features.shape[0], 1)
    if add_diff_phis:
        column_phis = PHIS_features.shape[1]
        for i in range(column_phis):
            if i != column_phis-1:
                phi_feature = PHIS_features[:, i].reshape(PHIS_features.shape[0], 1)
                subtracted_features = np.subtract(PHIS_features[:, i+1:], phi_feature)
                diff_features = np.hstack((diff_features, subtracted_features))
                
        diff_features = diff_features[:, 1:]
        features = np.hstack((features, diff_features))
    features = np.delete(features, params.PHIs_indices, axis=1)
    return features


def extract_from_dataset(y, tX, ids, condition, y_grouped, tX_grouped, ids_grouped):
    """
    Extract data from the dataset given some condition.

    Parameters
    ----------
    y: array
        The labels
    tX: 2D-matrix
        The features matrix
    ids: array
        The ids of the data points
    condition:
        The extracting condition
    y_grouped: array
        The grouped labels' array to append the next group to
    tX: 2D-matrix
        The grouped features matrix's array to append the next group to
    ids: array
        The grouped data points' ids array to append the next group to

    Returns
    -------
    y_grouped: 2D-array
        The labels grouped
    tX_grouped: object (array of arrays of data points)
        The features matrix grouped
    ids_grouped: 2D-array
        The data points' ids grouped
    """

    y_grouped.append(np.extract(condition, y))
    ids_grouped.append(np.extract(condition, ids))

    indices_to_take = np.extract(condition, range(len(tX)))
    tX_grouped.append(np.take(tX, indices_to_take, axis=0))

    return y_grouped, tX_grouped, ids_grouped


def split_in_groups_1(y, tX, ids, unwanted_value):
    """
    Split the dataset into groups according to the appearance of the unwanted value in each data point.

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

    y_grouped, tX_grouped, ids_grouped = [], [], []
    for i in range(max(indices) + 1):
        condition = indices == i
        y_grouped, tX_grouped, ids_grouped = extract_from_dataset(y, tX, ids, condition, y_grouped, tX_grouped,
                                                                  ids_grouped)
    return np.asarray(y_grouped), np.array(tX_grouped, dtype=object), np.asarray(ids_grouped), masks, counts


def split_in_groups_2(y, tX, ids, pri_jet_num_index, less_groups=False):
    """
        Split the dataset into groups according to the value of the feature PRI_jet_column (also called PRI_jet_num).

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
    y_grouped, tX_grouped, ids_grouped, masks, counts = [], [], [], [], []
    if less_groups:
        for i in range(params.PRI_jet_num_max_value):
            if i == params.PRI_jet_num_max_value - 1:
                condition = np.isin(tX.T[pri_jet_num_index], params.GROUPS_TO_MERGE)
            else:
                condition = tX.T[pri_jet_num_index] == i
            masks.append(condition)
            counts.append(np.sum(condition))
            y_grouped, tX_grouped, ids_grouped = extract_from_dataset(
                y, tX, ids, condition, y_grouped, tX_grouped, ids_grouped
            )
        return (
            np.asarray(y_grouped),
            np.array(tX_grouped, dtype=object),
            np.asarray(ids_grouped),
            masks,
            counts,
        )
    else:
        for i in range(params.PRI_jet_num_max_value + 1):
            condition = tX.T[pri_jet_num_index] == i
            masks.append(condition)
            counts.append(np.sum(condition))
            y_grouped, tX_grouped, ids_grouped = extract_from_dataset(
                y, tX, ids, condition, y_grouped, tX_grouped, ids_grouped
            )
        return (
            np.asarray(y_grouped),
            np.array(tX_grouped, dtype=object),
            np.asarray(ids_grouped),
            masks,
            counts,
        )


def additional_splitting(y_grouped, tX_grouped, ids_grouped, unwanted_value):
    y_grouped_new, tX_grouped_new, ids_grouped_new, masks_new, counts_new = [], [], [], [], []
    for i in range(len(tX_grouped)):
        y, tX, ids, masks, counts = split_in_groups_1(y_grouped[i], tX_grouped[i], ids_grouped[i], unwanted_value)
        for j in range(len(tX)):
            y_grouped_new.append(y[j])
            tX_grouped_new.append(tX[j])
            ids_grouped_new.append(ids[j])
            masks_new.append(masks[j])
            counts_new.append(counts[j])
    return y_grouped_new, tX_grouped_new, ids_grouped_new, masks_new, counts_new


def replace_unwanted_value_by_value(tX, unwanted_value, value):
    """
    In each feature, replace the unwanted value by the mean of the remaining values.

    Parameters
    ----------
    tX: 2D-array
        The features matrix
    unwanted_value: float
        The specific value we want to replace

    Returns
    -------
    The new matrix tX after replacing the unwanted value with the mean of the remaining features.
    """
    features = tX.T
    for i in range(len(features)):
        if value == 'mean':
            features[i][features[i] == unwanted_value] = np.mean(features[i][features[i] != unwanted_value])
        elif value == 'median':
            features[i][features[i] == unwanted_value] = np.median(features[i][features[i] != unwanted_value])
    return features.T


def replace_unwanted_value_by_value_grouped(tX_grouped, unwanted_value, value):
    """
    For each group and in each feature, replace the unwanted value by the mean of the remaining values in that features.

    Parameters
    ----------
    tX_grouped: list of arrays of data points
        The features matrix grouped
    unwanted_value: float
        The specific value we want to replace

    Returns
    -------
    The new matrix tX_grouped after replacing the unwanted value with the mean of the remaining features in each group.
    """
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(
            replace_unwanted_value_by_value(tX_grouped[i], unwanted_value, value)
        )
    return tX_grouped_new


def remove_invariable_features(tX):
    """
    Drop the features/columns that never change.

    Parameters
    ----------
    tX: 2D-array
        The features matrix

    Returns
    -------
    The features matrix cleaned of every invariable features.
    """

    features = tX.T
    stds = np.std(features, axis=1)
    indices = np.where(stds == 0)
    return np.delete(features, indices, 0).T


def remove_invariable_features_grouped(tX_grouped):
    """
    In each group of data points, drop the features/columns that never change.

    Parameters
    ----------
    tX_grouped: list of arrays of data points
        The features matrix grouped

    Returns
    -------
    tX_clean: array of arrays of data points
       The features matrix grouped and cleaned of every invariable features
    """

    tX_clean = []
    for i in range(len(tX_grouped)):
        tX_clean.append(remove_invariable_features(tX_grouped[i]))
    return tX_clean


def standardize(tX):
    """
    Standardize each feature/column.

    Parameters
    ----------
    tX: 2D-array
        The features matrix

    Returns
    -------
    The features matrix standardized
    """
    features = tX.T
    features_len = len(features)
    means = np.reshape(np.mean(features, axis=1), [features_len, 1])
    stds = np.reshape(np.std(features, axis=1), [features_len, 1])
    features_std = (features - means) / stds
    return features_std.T


def standardize_grouped(tX_grouped):
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
        tX_clean_std.append(standardize(tX_grouped[i]))
    return tX_clean_std


def replace_outliers_by_threshold(tX, threshold, outlier_value):
    """
    In each feature, replace the outliers by the appropriate value.
    
    Parameters
    ----------
    tX: array of data points 
        The features matrix 
    threshold: integer
        The parameter that defines how the value should be close to the mean
    
    Returns
    -------
    tX: array of data points
        The features matrix with less outliers
    """
    
    for j in range(tX.shape[1]):
        col = tX[:, j]
        values, indices = np.unique(col, return_index=True)
        data = zip(values, indices)
        values_mean = np.mean(values)
        values_std = np.std(values)
        cut_off = threshold * values_std
        lower, upper = values_mean - cut_off, values_mean + cut_off
        outliers = []
        other_values = []
        for (x, y) in data:
            if x < lower or x > upper:
                outliers.append((x, y))
            else:
                other_values.append((x, y))
        lower_mean = np.mean(np.asarray(other_values)[other_values <= values_mean])
        upper_mean = np.mean(np.asarray(other_values)[other_values >= values_mean])
        for v, index in outliers:
            if outlier_value == 'clip':
                if v < values_mean:
                    tX[index, j] = lower
                else:
                    tX[index, j] = upper
            elif outlier_value == 'mean':
                tX[index, j] = values_mean
            elif outlier_value == 'upper_lower_mean':
                if v < values_mean:
                    tX[index, j] = lower_mean
                else:
                    tX[index, j] = upper_mean
    return tX


def replace_outliers_grouped(tX_grouped, threshold, outlier_value):
    """
    In each group of data points, replace the outliers by the appropriate value.

    Parameters
    ----------
    tX_grouped: array of arrays of data points
        The features matrix grouped
    threshold: integer
        The parameter that defines how the value should be close to the mean

    Returns
    -------
    tX_grouped: array of arrays of data points
        The features matrix grouped with less outliers
    """
    for i in range(len(tX_grouped)):
        tX_i = tX_grouped[i]
        tX_grouped[i] = replace_outliers_by_threshold(tX_i, threshold, outlier_value)
    return tX_grouped
