import numpy as np
import matplotlib.pyplot as plt

import params
import proj1_helpers as helpers
import implementations as impl


def locally_predict(tX, y, counts, implementation=params.IMPLEMENTATION, group=params.GROUP, ratio=params.RATIO,
                    cross_validation=params.CROSS_VALIDATION, k=params.K, max_iter=params.MAX_ITERS, gamma=params.GAMMA,
                    decreasing_gamma=params.DECREASING_GAMMA, log_lambda=params.LOG_LAMBDA, ridge_lambda=params.RIDGE_LAMBDA):
    """ 
    Predict locally the accruacy of a model according to various parameters.
    
    Parameters
    ----------
    
    tX: array
        The feature matrix
    y:  array
        The output array
    counts: list
        The list containing the number of points for each groups
    implementation: integer
        The integer that determine which type of regression is used (0 = LEAST_SQUARES, 1 = RIDGE_REGRESSION,
        3 = LOGISTIC_REGRESSION)
    group: boolean
        Either the dataset is divided into groups or not
    ratio: real number
        The ratio used in cross-validation
    cross_validation: boolean
        Either the cross validation is performed or not
    max_iter: integer
        Number of iterration for the gradient descent (if performed)
    gamma: real number
        gamma parameter for GD or SGD
    log_lamda: real number
        lambda parameter for logistic regression
    ridge_lambda: real number
        lambda parameter for ridge regression 
        
        
    Return
    ------
    accuarcy: float or list
    """    
    if group:
        if cross_validation:
            return cross_validate_grouped(tX, y, ratio, k, implementation, max_iter, gamma, decreasing_gamma, log_lambda,
                                   ridge_lambda, counts)
        else:
            tX_train_grouped, y_train_grouped, tX_sub_test_grouped, y_sub_test_grouped = separate_data_grouped(tX, y, ratio)
            log_initial_ws = []
            for i in range(len(tX)):
                log_initial_ws.append(np.repeat(0, tX_train_grouped[i].shape[1]))
            optimal_ws = find_optimal_ws_grouped(tX_train_grouped, y_train_grouped, implementation, log_initial_ws,
                                                 max_iter, gamma, decreasing_gamma, log_lambda, ridge_lambda)
            y_pred_grouped, y_pred_clipped_grouped = helpers.predict_labels_grouped(optimal_ws, tX_sub_test_grouped,
                                                                                    implementation)
            return compare_labels_grouped(y_pred_grouped, y_pred_clipped_grouped, y_sub_test_grouped, implementation, counts)
    else:
        if cross_validation:
            return cross_validate(tX, y, ratio, k, implementation, max_iter, gamma, decreasing_gamma, log_lambda, ridge_lambda, counts)
        else:
            tX_train, y_train, tX_sub_test, y_sub_test = separate_data(tX, y, ratio)
            optimal_w = find_optimal_w(tX_train, y_train, implementation, np.repeat(0, tX_train.shape[1]), max_iter,
                                       gamma, decreasing_gamma, log_lambda, ridge_lambda)
            y_pred, y_pred_clipped = helpers.predict_labels(optimal_w, tX_sub_test, implementation)
            return compare_labels(y_pred, y_pred_clipped, y_sub_test, implementation, counts)


def separate_data(tX, y, ratio):
    """
    Separate the dataset according to the ration
    
    Parameters 
    ----------
    
    tX: array
        The feature matrix
    y: array
        The output
    
    Return
    ------
    
    tX_train: array
        The training set
    y_train: array
        The training label set 
    tX_sub_test: array
        The test set
    y_sub_test: array
        The test label set
    """    
    index = int(len(y) * ratio)
    tX_train = tX[:index]
    y_train = y[:index]
    tX_sub_test = tX[index:]
    y_sub_test = y[index:]
    return tX_train, y_train, tX_sub_test, y_sub_test


def separate_data_grouped(tX_grouped, y_grouped, ratio):
    """
    Perform separate_data on a list of dataset
    
    Parameters 
    ----------
    
    tX_grouped: list
        The list of feature matrices
    y_grouped: list
        The list of label arrays
    ratio: integer
        Ration according to which the dataset will be separated
    
    Return
    ------
    
    tX_train_grouped: list 
        The training sets according to groups
    y_train_grouped: list
        The training label sets according to groups 
    tX_sub_test_grouped: list
        The testing set according to groups
    y_sub_test_grouped: list
        The testing label set according to groups
    """    
    tX_train_grouped = []
    y_train_grouped = []
    tX_sub_test_grouped = []
    y_sub_test_grouped = []
    for i in range(len(tX_grouped)):
        tX_train, y_train, tX_sub_test, y_sub_test = separate_data(tX_grouped[i], y_grouped[i], ratio)
        tX_train_grouped.append(tX_train)
        y_train_grouped.append(y_train)
        tX_sub_test_grouped.append(tX_sub_test)
        y_sub_test_grouped.append(y_sub_test)
    return tX_train_grouped, y_train_grouped, tX_sub_test_grouped, y_sub_test_grouped


def find_optimal_w(tX, y, implementation, log_initial_w, log_max_iters, log_gamma, decreasing_gamma, log_regulator, ridge_lambda):
    """
    Find the optimal weights by training the data set
    
    Parameters 
    ----------
    
    tX: array
        The feature matrices
    y: array
        The output
    log_initial_w: array
        inital weights in order to perform GD or SGD
    log_max_iters: integer
        number of iterations to perform GD or SGD
    log_gamma: float
        gamma parameter to perform GD or SGD
    log_regulator: float
        lambda to perform logistic regression
    ridge_lambda: float
        lambda to perform ridge regression
      
    Return
    ------
    
    optimal_w = array
        Optimal weights.

    """    
    optimal_w = None
    if implementation == 0:
        optimal_w, _ = impl.least_squares(y, tX)
    if implementation == 1:
        optimal_w, _ = impl.ridge_regression(y, tX, ridge_lambda)
    if implementation == 2:
        optimal_w, _ = impl.reg_logistic_regression(y, tX, log_regulator, log_initial_w, log_max_iters, log_gamma, decreasing_gamma)
    return optimal_w


def find_optimal_ws_grouped(tX_grouped, y_grouped, implementation, log_initial_w, log_max_iters, log_gamma, decreasing_gamma,
                            log_regulator, ridge_lambda):
    """
    Perform find_optimal_w on list of data matrices
    
    Parameters 
    ----------
    
    tX_grouped: list
        List of feature matrices
    y: array
        The output
    log_initial_w: array
        inital weights in order to perform GD or SGD
    log_max_iters: integer
        number of iterations to perform GD or SGD
    log_gamma: float
        gamma parameter to perform GD or SGD
    log_regulator: float
        lambda to perform logistic regression
    ridge_lambda: float
        lambda to perform ridge regression
      
    Return
    ------
    
    optimal_w = array
        Optimal weights.

    """        
    optimal_ws = []
    for i in range(len(tX_grouped)):
        optimal_ws.append(find_optimal_w(tX_grouped[i], y_grouped[i], implementation, log_initial_w[i], log_max_iters,
                                         log_gamma, decreasing_gamma, log_regulator, ridge_lambda))
        print('\t\tFound optimal w for group {}.'.format(i))
    print('\tOptimal ws found.')
    return optimal_ws


def compare_labels(y_pred, y_pred_clipped, y_sub_test, implementation, count, group_number=0):
    """
    Compare the labels from test set to the one issued with the optimal weights found
    
    Parameters 
    ----------
    
    y_pred_clipped: array
        The output predicted with the optimal weights (-1 or 1)
    y_sub_test: array
        The outputs of the test subset
      
    Return
    ------
    
    accuracy = float
        Accuracy.

    """    
    comparison = np.abs(y_pred_clipped + y_sub_test)
    unique, counts = np.unique(comparison, return_counts=True)
    accuracy = counts[1] / len(comparison)
    # if params.DEBUG:
        # print('\t\tGroup {} ({} data points)'.format(group_number, count))
        # print('\t\t\tImplementation = {}'.format(implementation))
        # print('\t\t\tAccuracy = {}'.format(accuracy))
        # plt.figure()
        # plt.hist(y_pred, bins='auto')
        # plt.title('Group {}'.format(group_number))
        # plt.xlabel('y prediction')
        # plt.ylabel('Number of data points with this prediction')
    return accuracy


def compare_labels_grouped(y_pred_grouped, y_pred_clipped_grouped, y_sub_test_grouped, implementation, counts):
    """
    Perform compage_labels of a list of data matrices
    
    Parameters 
    ----------
    
    y_pred_grouped: array
        The list of outputs predicted with the optimal weights (-1 or 1)
    y_sub_test_grouped: array
        The list of outputs of the test subset
      
    Return
    ------
    
    accuracies = list
        List of accuracy for each group.

    """    
    accuracies = []
    for i in range(len(y_pred_clipped_grouped)):
        accuracies.append(compare_labels(y_pred_grouped[i], y_pred_clipped_grouped[i], y_sub_test_grouped[i],
                                         implementation, counts[i], i))
    print('\nOverall accuracy = {}'.format(np.average(accuracies, weights=counts)))
    return accuracies


def cross_validate(tX, y, ratio, k, implementation, max_iter, gamma, decreasing_gamma, log_lambda, ridge_lambda, count, group_number=0):
    """
    Cross validation: Separate the data according to the ratio and perform the local training/testing.  
    
    Parameters 
    ----------
    
    tX: array
        The feature matrix
    y: array
        The output
    implementation: integer
        Determines which regression is used
    max_iter: integer
        Number of interation for GD or SGD
    gamma: float
        gamma parameter for GD or SGD
    log_lambda: float
        lambda parameter for logistic regression
    rigde_lambda: float
        lambda parameter for ridge regression
    count: integer
        Cardinality of the output
      
    Return
    ------
    
    accuracies = float
        Accuracy.

    """    
    n_parts = int(1/(1-ratio))
    tX_split = np.asarray(np.array_split(tX, n_parts, axis=0))
    y_split = np.array_split(y, n_parts, axis=0)
    indices_to_choose = np.arange(n_parts)
    accuracies = []

    for i in range(k):
        # print('Cross validation {}'.format(i))
        chosen_index = np.random.choice(indices_to_choose)
        condition = np.full(n_parts, True)
        condition[chosen_index] = False
        indices_to_choose = indices_to_choose[indices_to_choose != chosen_index]

        tX_train = np.compress(condition, tX_split, axis=0)
        tX_train = np.vstack(tX_train)

        y_train = np.compress(condition, y_split, axis=0)
        y_train = np.hstack(y_train)

        tX_sub_test = np.compress(~condition, tX_split, axis=0)[0]

        y_sub_test = np.compress(~condition, y_split, axis=0)[0]

        optimal_w = find_optimal_w(tX_train, y_train, implementation, np.repeat(0, tX_train.shape[1]), max_iter,
                                   gamma, decreasing_gamma, log_lambda, ridge_lambda)
        y_pred, y_pred_clipped = helpers.predict_labels(optimal_w, tX_sub_test, implementation)
        accuracies.append(compare_labels(y_pred, y_pred_clipped, y_sub_test, implementation, count, group_number=group_number))

    mean_accuracy = np.mean(accuracies)
    print('\t\tGroup {} ({} data points)'.format(group_number, count))
    print('\t\t\tImplementation = {}'.format(implementation))
    print('\t\t\tAccuracy = {}'.format(mean_accuracy))
    return mean_accuracy


def cross_validate_grouped(tX_grouped, y_grouped, ratio, k, implementation, max_iter, gamma, decreasing_gamma, log_lambda, ridge_lambda, counts):
    """
    Perform cross validation on list of data matrices 
    
    Parameters 
    ----------
    
    tX_grouped: list
        The list of data matrices
    y_grouped: array
        The list of output
    implementation: integer
        Determines which regression is used
    max_iter: integer
        Number of interation for GD or SGD
    gamma: float
        gamma parameter for GD or SGD
    log_lambda: float
        lambda parameter for logistic regression
    rigde_lambda: float
        lambda parameter for ridge regression
    counts: list 
        List of cardinality of the output of each group 
      
    Return
    ------
    
    accuracies = list
        List of accuracies.

    """        
    accuracies = []
    for i in range(len(tX_grouped)):
        accuracies.append(cross_validate(tX_grouped[i], y_grouped[i], ratio, k, implementation, max_iter, gamma, decreasing_gamma, log_lambda, ridge_lambda, counts[i], i))
    print('\nOverall accuracy = {}'.format(np.average(accuracies, weights=counts)))
