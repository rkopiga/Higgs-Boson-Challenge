import numpy as np
import params
import proj1_helpers as helper

"""
The preprocessing functions.
"""


def preprocess(
        y,
        tX,
        ids,
        shuffle=params.SHUFFLE_DATA,
        remove_phis=params.REMOVE_PHIS,
        unwanted_value=params.UNWANTED_VALUE,
        pri_jet_num_index=params.PRI_jet_num_index,
        group=params.GROUP,
        group_1=params.GROUP_1,
        group_2=params.GROUP_2,
        group_2_additional_splitting=params.GROUP_2_ADDITIONAL_SPLITTING,
        less_groups=params.LESS_GROUPS,
        replace_unwanted_value=params.REPLACE_UNWANTED_VALUE,
        value=params.VALUE,
        remove_inv_features=params.REMOVE_INV_FEATURES,
        std=params.STD,
        replace_outliers=params.REPLACE_OUTLIERS,
        outlier_value=params.OUTLIER_VALUE,
        threshold=params.THRESHOLD,
        remove_duplicate_features=params.REMOVE_DUPLICATE_FEATURES):
    """
    Preprocess the dataset.

    Parameters
    ----------
    y: array
        The labels
    tX: array
        The features matrix
    ids: array
        The ids of the data points
    shuffle: boolean
        Rather we shuffle the data points or not
    remove_phis: boolean
        Rather we remove the phi-features or not
    unwanted_value: int
        The value indicating a missing/not-supposed-to-be-there measurement
    pri_jet_num_index: int
        The index at which the PRI_jet_num is in the features matrix
    group: boolean
        Rather we organize the dataset in groups or not
    group_1: boolean
        Rather we split the dataset according to the appearance of UNWANTED_VALUE or not
    group_2: boolean
        Rather we split the dataset according to the value of PRI_jet_num or not
    group_2_additional_splitting: boolean
        When splitting with group_2, rather we split each group again in 2 groups according to DER_mass_MMC or not
    less_groups: boolean
        When splitting with group_2, rather we split the dataset in 3 groups instead of 4 or not
    replace_unwanted_value: boolean
        Rather we replace all the unwanted values by the mean of the remaining values in each feature or not
    value: str
        Indicating with which value to replace the UNWANTED_VALUE (mean, median, etc.)
    remove_inv_features: boolean
        Rather we remove all the invariable features or not
    std: boolean
        Rather we standardize each feature in each group or not
    replace_outliers: boolean
        Rather we replace the outliers in each group or not
    outlier_value: str
        Indicating with which value to replace the outliers (clip, mean, etc.)
    threshold: integer
        Parameter that defines an outlier
    remove_duplicate_features: boolean
        Rather we remove the duplicate features at the end or not

    Returns
    -------
    y: list/array
        The labels (with groups or not)
    tX: list/array
        The (list of) features matrix
    ids: list/array
        The ids of the data points (with groups or not)
    masks: list/array or None
        The features of each group expressed in terms of 1 (which means there is an UNWANTED_VALUE at this position)
        and 0 (no UNWANTED_VALUE at this position), or None depending on the chosen split function
    counts: array (or int)
        The number of data points belonging to each group depending on the chosen split function (or the number of data
        points if no grouping is used)
    """
    print('\tPreprocessing...')
    print('REMOVE PHIS :',remove_phis)
    print('REMOVE INV FEATURES', remove_inv_features)
    print('STD', std)
    print('REPLACE OUTLIERS', replace_outliers)
    masks = None
    counts = None

    if shuffle:
        y, tX, ids = shuffle_data(y, tX, ids)

    if remove_phis:
        tX = handle_angle_phis(tX)
        pri_jet_num_index = params.PRI_jet_num_new_index
    
    if group:
        if group_1:
            y, tX, ids, masks, counts = split_in_groups_1(y, tX, ids, unwanted_value)
        elif group_2:
            y, tX, ids, masks, counts = split_in_groups_2(y, tX, ids, pri_jet_num_index, less_groups)
            # check_uniqueness_in_group(tX, unwanted_value)
            if group_2_additional_splitting:
                y, tX, ids, masks, counts = additional_splitting(y, tX, ids, unwanted_value)
        if remove_inv_features:
            tX = remove_invariable_features_grouped(tX)
        if replace_unwanted_value:
            tX = replace_unwanted_value_by_value_grouped(tX, unwanted_value, value)
        if std:
            tX = standardize_grouped(tX)
        if replace_outliers:
            tX = replace_outliers_grouped(tX, threshold, outlier_value)
        if remove_duplicate_features:
            tX = helper.remove_duplicate_columns_grouped(tX)
    else:
        counts = len(y)
        if remove_inv_features:
            tX = remove_invariable_features(tX)
        if replace_unwanted_value:
            tX = replace_unwanted_value_by_value(tX, unwanted_value, value)
        if std:
            tX = standardize(tX)
        if replace_outliers:
            tX = replace_outliers_by_threshold(tX, threshold, outlier_value)
        if remove_duplicate_features:
            tX = helper.remove_duplicate_columns(tX)
    print('\tPreprocessing ok.')
    return y, tX, ids, masks, counts


def check_uniqueness_in_group(tX_grouped, unwanted_value):
    """
    Check that according to the unwanted_value, we cannot split the groups more than they currently are.

    Parameters
    ----------
    tX_grouped: list
        The list of features matrix (one for each group)
    unwanted_value: int
        The value based on which we will say if we can split the groups even more

    Returns
    -------
    None
    """
    masks_check = []
    counts_check = []
    for i in range(len(tX_grouped)):
        unwanted_value_check = 1 * (tX_grouped[i] == unwanted_value)
        masks_and_counts = np.unique(unwanted_value_check, return_counts=True, axis=0)
        masks_check.append(masks_and_counts[0])
        counts_check.append(masks_and_counts[1])
    print(masks_check)
    print(counts_check)
    return None


def shuffle_data(y, tX, ids):
    """
    Shuffle the dataset.

    Parameters
    ----------
    y: array
        The labels
    tX: array
        The features matrix
    ids: array
        The ids of the data points

    Returns
    -------
    y_shuffled: array
        The labels shuffled
    tX_shuffled: array
        The features matrix shuffled
    ids_shuffled: array
        The ids of the data points, shuffled
    """
    y = y.reshape(y.shape[0], 1)
    ids = ids.reshape(ids.shape[0], 1)
    model_data = np.hstack((tX, y, ids))
    np.random.shuffle(model_data)
    ids_shuffled = model_data[:, model_data.shape[1] - 1]
    y_shuffled = model_data[:, model_data.shape[1] - 2]
    tX_shuffled = model_data[:, :model_data.shape[1] - 2]
    return y_shuffled, tX_shuffled, ids_shuffled


def handle_angle_phis(tX):
    """
    Delete the phi-features from the features matrix.

    Parameters
    ----------
    tX: array
        The features matrix

    Returns
    -------
    new_tX: array
        The new features matrix
    """
    new_tX = tX
    new_tX = np.delete(new_tX, params.PHIs_indices, axis=1)
    return new_tX


def extract_from_dataset(y, tX, ids, condition, y_grouped, tX_grouped, ids_grouped):
    """
    Extract data from the dataset given some condition.

    Parameters
    ----------
    y: array
        The labels
    tX: array
        The features matrix
    ids: array
        The ids of the data points
    condition: array
        The extracting condition
    y_grouped: array
        The labels grouped
    tX_grouped: list
        The list of features matrices to append the next group to
    ids_grouped: list
        The list of ids of data points to append the next group to

    Returns
    -------
    y_grouped: list
        The labels grouped
    tX_grouped: list
        The list of features matrix
    ids_grouped: list
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
    tX: array
        The features matrix
    ids: array
        The ids of the data points
    unwanted_value:
        The value according to which we form the groups

    Returns
    -------
    y_grouped: list
        The labels grouped
    tX_grouped: list
        The features matrix grouped
    ids_grouped: list
        The data points' ids grouped
    masks: array
        The features of each group expressed in terms of 1 (means there is a UNWANTED_VALUE at this position) and
        0 otherwise
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
    Split the dataset into groups according to the value of the feature PRI_jet_num.

    Parameters
    ----------
    y: array
        The labels
    tX: array
        The features matrix
    ids: array
        The ids of the data points
    pri_jet_num_index: int
        The index at which the PRI_jet_num is in the features matrix
    less_groups: boolean
        Rather we form 3 groups instead of 4 (by merging the last 2)

    Returns
    -------
    y_grouped: list
        The labels grouped
    tX_grouped: list
        The features matrix grouped
    ids_grouped: list
        The data points' ids grouped
    masks: array
        The features of each group expressed in terms of 1 (means there is a UNWANTED_VALUE at this position) and
        0 otherwise
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
    """
    Split each group of an existing grouping with the split_in_groups_1 function.

    Parameters
    ----------
    y_grouped: list
        The labels grouped
    tX_grouped: list
        The features matrix grouped
    ids_grouped: list
        The ids of the data points, grouped
    unwanted_value:
        The value according to which we form the groups

    Returns
    -------
    y_grouped_new: list
        The labels grouped
    tX_grouped_new: list
        The features matrix grouped
    ids_grouped_new: list
        The data points' ids grouped
    masks_new: array
        The features of each group expressed in terms of 1 (means there is a UNWANTED_VALUE at this position) and
        0 otherwise
    counts_new: array
        The number of data points belonging to each group
    """
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
    Replace the unwanted value by the mean or median (according to the parameter value)
    of the remaining values in each feature.

    Parameters
    ----------
    tX: array
        The features matrix
    unwanted_value: int
        The specific value we want to replace
    value: str
        Indicating with which value to replace the UNWANTED_VALUE (mean or median)

    Returns
    -------
    new_tX: array
        The new features matrix
    """
    features = tX.T
    for i in range(len(features)):
        if value == 'mean':
            features[i][features[i] == unwanted_value] = np.mean(features[i][features[i] != unwanted_value])
        elif value == 'median':
            features[i][features[i] == unwanted_value] = np.median(features[i][features[i] != unwanted_value])
    new_tX = features.T
    return new_tX


def replace_unwanted_value_by_value_grouped(tX_grouped, unwanted_value, value):
    """
    For each group, replace the unwanted value by the mean of the remaining values in each feature.

    Parameters
    ----------
    tX_grouped: list
        The list of features matrix
    unwanted_value: int
        The specific value we want to replace
    value: str
        Indicating with which value to replace the UNWANTED_VALUE (mean or median)

    Returns
    -------
    new_tX: list
        The new list of features matrices
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
    tX: array
        The features matrix

    Returns
    -------
    new_tX: array
        The new features matrix
    """

    features = tX.T
    stds = np.std(features, axis=1)
    indices = np.where(stds == 0)
    new_tX = np.delete(features, indices, 0).T
    return new_tX


def remove_invariable_features_grouped(tX_grouped):
    """
    Drop the features/columns that never change for each group.

    Parameters
    ----------
    tX_grouped: list
        The list of features matrices

    Returns
    -------
    tX_clean: list
       The cleaned list of features matrices
    """

    new_tX_grouped = []
    for i in range(len(tX_grouped)):
        new_tX_grouped.append(remove_invariable_features(tX_grouped[i]))
    return new_tX_grouped


def standardize(tX):
    """
    Standardize each feature of the feature matrix.

    Parameters
    ----------
    tX: array
        The features matrix

    Returns
    -------
    new_tX: array
        The new features matrix
    """
    features = tX.T
    features_len = len(features)
    means = np.reshape(np.mean(features, axis=1), [features_len, 1])
    stds = np.reshape(np.std(features, axis=1), [features_len, 1])
    features_std = (features - means) / stds
    new_tX = features_std.T
    return new_tX


def standardize_grouped(tX_grouped):
    """
    Standardize each feature of the feature matrix for each group.

    Parameters
    ----------
    tX_grouped: list
        The list of features matrices

    Returns
    -------
    tX_clean_std: list
       The new list of features matrices standardized
    """

    new_tX_grouped = []
    for i in range(len(tX_grouped)):
        new_tX_grouped.append(standardize(tX_grouped[i]))
    return new_tX_grouped


def replace_outliers_by_threshold(tX, threshold, outlier_value):
    """
    Replace the outliers by the appropriate value in each feature.
    
    Parameters
    ----------
    tX: array
        The features matrix 
    threshold: float
        The parameter that defines how the value should be close to the mean
    outlier_value: str
        Indicating with which value to replace the UNWANTED_VALUE (clip, mean, or upper_lower_mean)
    
    Returns
    -------
    tX: array
        The new features matrix with less outliers
    """

    new_tX = tX
    for j in range(new_tX.shape[1]):
        col = new_tX[:, j]
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
                    new_tX[index, j] = lower
                else:
                    new_tX[index, j] = upper
            elif outlier_value == 'mean':
                new_tX[index, j] = values_mean
            elif outlier_value == 'upper_lower_mean':
                if v < values_mean:
                    new_tX[index, j] = lower_mean
                else:
                    new_tX[index, j] = upper_mean
    return new_tX


def replace_outliers_grouped(tX_grouped, threshold, outlier_value):
    """
    Replace the outliers by the appropriate value in each feature, in each group.

    Parameters
    ----------
    tX_grouped: list
        The list of features matrices
    threshold: float
        The parameter that defines how the value should be close to the mean
    outlier_value: str
        Indicating with which value to replace the UNWANTED_VALUE (clip, mean, or upper_lower_mean)

    Returns
    -------
    tX_grouped: list
        The new features matrix with less outliers
    """
    new_tX_grouped = tX_grouped
    for i in range(len(new_tX_grouped)):
        new_tX_grouped[i] = replace_outliers_by_threshold(tX_grouped[i], threshold, outlier_value)
    return new_tX_grouped
