import numpy as np
import preprocessing as prep
import local_prediction as pred
import feature_engineering as f_e
import params
import proj1_helpers as helpers

"""
The big grid search file: used to find the best parameters for each group.
"""


ridge_lambda_range = [10**-(i+1) for i in range(15)]
max_iter_range = [25, 30, 50]
gamma_range = [2**-(i+1) for i in range(15)]
log_lambda_range = [2 ** -(i + 1) for i in range(15)]
true_false = [True, False]


def ridge_regression_grid_search(tX, y, ridge_lambda_range):
    """
    Do a local prediction on tX and y for each lambda in ridge_lambda_range using ridge regression.

    Parameters
    ----------
    tX: array
        The features matrix
    y: array
        The labels
    ridge_lambda_range: array
        The lambdas we want to try

    Returns
    -------
    None
    """
    accuracies = []
    for ridge_lambda in ridge_lambda_range:
        accuracies.append(pred.locally_predict(tX, y, counts=len(tX), implementation=1, group=False, ridge_lambda=ridge_lambda))
    argmax = np.flip(np.argsort(accuracies), axis=0)[0]
    print(ridge_lambda_range[argmax], accuracies[argmax])


def reg_logistic_regression_grid_search(tX, y, group_number, max_iter_range, gamma_range, log_lambda_range):
    """
    Do a local prediction on tX and y for each max_iter in max_iter_range, each gamma in gamma_range, and each lambda
    in log_lambda_range, using regularized logistic regression.

    Parameters
    ----------
    tX: array
        The features matrix
    y: array
        The labels
    group_number: int
        The index of the current group
    max_iter_range: array
        The max_iter parameters we want to try
    gamma_range: array
        The gamma parameters we want to try
    log_lambda_range: array
        The log_lambda parameters we want to try

    Returns
    -------
    None
    """
    accuracies = []
    print('Group {}'.format(group_number))
    for max_iter in max_iter_range:
        for gamma in gamma_range:
            for log_lambda in log_lambda_range:
                accuracies.append(pred.locally_predict(tX, y, counts[group_number], implementation=2, group=False,
                                                       max_iter=max_iter, gamma=gamma, log_lambda=log_lambda))
    argmax = np.flip(np.argsort(accuracies), axis=0)[0]
    print(max_iter_range[argmax], gamma_range[argmax], log_lambda_range[argmax], accuracies[argmax])


if __name__ == "__main__":
    y, tX, ids = helpers.load_csv_data(params.DATA_TRAIN_PATH)
    for replace_unwanted_value in true_false:
        for std in true_false:
            print('\t\treplace_unwanted_value = {}'.format(replace_unwanted_value))
            print('\t\tstd = {}'.format(std))
            y_grouped_preprocessed, tX_grouped_preprocessed, ids_grouped_preprocessed, masks, counts = \
                prep.preprocess(y, tX, ids, std=std, replace_unwanted_value=replace_unwanted_value)
            for ones_column in true_false:
                for feature_multiplication in true_false:
                    print('\t\tones_column = {}'.format(ones_column))
                    print('\t\tfeature_multiplication = {}'.format(feature_multiplication))
                    tX_improved = f_e.feature_engineer(tX_grouped_preprocessed[0], group=False, one_column=ones_column,
                                                       polynomial_multiplication=feature_multiplication)

                    ridge_regression_grid_search(tX_improved, y_grouped_preprocessed[0], ridge_lambda_range)
                    #reg_logistic_regression_grid_search(tX_improved, y_grouped_preprocessed[0], 0, max_iter_range, gamma_range, log_lambda_range)
