import numpy as np
import matplotlib.pyplot as plt

import params
import proj1_helpers as helpers
import implementations as impl


def locally_predict(tX, y, counts, implementation=params.IMPLEMENTATION, group=params.GROUP, ratio=params.RATIO):
    if group:
        tX_train_grouped, y_train_grouped, tX_sub_test_grouped, y_sub_test_grouped = separate_data_grouped(tX, y, ratio)
        log_initial_ws = []
        for i in range(len(tX)):
            log_initial_ws.append(np.repeat(0, tX_train_grouped[i].shape[1]))
        optimal_ws = find_optimal_ws_grouped(tX_train_grouped, y_train_grouped, implementation, log_initial_ws,
                                             params.MAX_ITERS, params.GAMMA, params.REGULATOR, params.RIDGE_LAMBDA)
        y_pred_grouped, y_pred_clipped_grouped = helpers.predict_labels_grouped(optimal_ws, tX_sub_test_grouped, implementation)
        return compare_labels_grouped(y_pred_grouped, y_pred_clipped_grouped, y_sub_test_grouped, implementation, counts)
    else:
        tX_train, y_train, tX_sub_test, y_sub_test = separate_data(tX, y, ratio)
        optimal_w = find_optimal_w(tX, y, implementation, np.repeat(0, tX_train.shape[1]), params.MAX_ITERS, params.GAMMA, params.REGULATOR,
                                   params.RIDGE_LAMBDA)
        y_pred, y_pred_clipped = helpers.predict_labels(optimal_w, tX_sub_test, implementation)
        return compare_labels(y_pred, y_pred_clipped, y_sub_test, implementation, counts)


def separate_data(tX, y, ratio):
    index = int(len(y) * ratio)
    tX_train = tX[:index]
    y_train = y[:index]
    tX_sub_test = tX[index:]
    y_sub_test = y[index:]
    return tX_train, y_train, tX_sub_test, y_sub_test


def separate_data_grouped(tX_grouped, y_grouped, ratio):
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


def find_optimal_w(tX, y, implementation, log_initial_w, log_max_iters, log_gamma, log_regulator, ridge_lambda):
    optimal_w = None
    if implementation == 0:
        optimal_w, _ = impl.least_squares(y, tX)
    if implementation == 1:
        optimal_w, _ = impl.ridge_regression(y, tX, ridge_lambda)
    if implementation == 2:
        optimal_w, _ = impl.logistic_regression_GD(y, tX, log_initial_w, log_max_iters, log_gamma, log_regulator)
    return optimal_w


def find_optimal_ws_grouped(tX_grouped, y_grouped, implementation, log_initial_w, log_max_iters, log_gamma,
                            log_regulator, ridge_lambda):
    optimal_ws = []
    for i in range(len(tX_grouped)):
        optimal_ws.append(find_optimal_w(tX_grouped[i], y_grouped[i], implementation, log_initial_w[i], log_max_iters,
                                         log_gamma, log_regulator, ridge_lambda))
        print('\t\tFound optimal w for group {}.'.format(i))
    print('\tOptimal ws found.')
    return optimal_ws


def compare_labels(y_pred, y_pred_clipped, y_sub_test, implementation, count, group_number=0):
    comparison = np.abs(y_pred_clipped + y_sub_test)
    unique, counts = np.unique(comparison, return_counts=True)
    accuracy = counts[1] / len(comparison)
    if params.DEBUG:
        print('\t\tGroup {} ({} data points)'.format(group_number, count))
        print('\t\t\tImplementation = {}'.format(implementation))
        print('\t\t\tAccuracy = {}'.format(accuracy))
        # plt.figure()
        # plt.hist(y_pred, bins='auto')
        # plt.title('Group {}'.format(group_number))
        # plt.xlabel('y prediction')
        # plt.ylabel('Number of data points with this prediction')
    return accuracy


def compare_labels_grouped(y_pred_grouped, y_pred_clipped_grouped, y_sub_test_grouped, implementation, counts):
    accuracies = []
    for i in range(len(y_pred_clipped_grouped)):
        accuracies.append(compare_labels(y_pred_grouped[i], y_pred_clipped_grouped[i], y_sub_test_grouped[i],
                                         implementation, counts[i], i))
    print('\nOverall accuracy = {}'.format(np.average(accuracies, weights=counts)))
    return accuracies
