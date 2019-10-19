from proj1_helpers import *
from params import *


def feature_engineering(tX,
                        group=GROUP,
                        polynomial_expansion=FEATURE_EXPANSION,
                        degree=DEGREE,
                        polynomial_multiplication=FEATURE_MULTIPLICATION,
                        one_column=ONE_COLUMN):
    if group:
        if polynomial_expansion:
            tX = feature_expansion_grouped(tX, degree)
        if polynomial_multiplication:
            tX = feature_multiplication_grouped(tX)
        if one_column:
            tX = add_ones_column_grouped(tX)
    else:
        if polynomial_expansion:
            tX = feature_expansion(tX, degree)
        if polynomial_multiplication:
            tX = feature_multiplication(tX)
        if one_column:
            tX = add_ones_column(tX)
    print('\tFeature engineering ok.')
    return tX


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=2 up to j=degree."""
    a = [np.power(x, d) for d in range(2, degree+1)]
    return np.asarray(a)


def feature_expansion(tX, degree):
    for feature_index in range(tX.shape[1]):
        feature = tX[:, feature_index]
        expanded_feature = build_poly(feature, degree).T
        tX = np.hstack((tX, expanded_feature))
    return tX


def feature_expansion_grouped(tX_grouped, degree):
    tX_expanded = []
    for i in range(len(tX_grouped)):
        tX_expanded.append(feature_expansion(tX_grouped[i], degree))
    return tX_expanded


def feature_multiplication(tX):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    new_tX = tX
    for i in range(tX.shape[1]):
        col = tX[:, i].reshape(tX.shape[0], 1)
        tX_concat = np.multiply(tX[:, i:], col)
        new_tX = np.hstack((new_tX, tX_concat))
    return new_tX


def feature_multiplication_grouped(tX_grouped):
    new_tX_grouped = []
    for i in range(len(tX_grouped)):
        new_tX_grouped.append(feature_multiplication(tX_grouped[i]))
    return new_tX_grouped


def add_ones_column(tX):
    len_tX = len(tX)
    ones = np.reshape(np.ones(len_tX), [len_tX, 1])
    return np.hstack((ones, tX))


def add_ones_column_grouped(tX_grouped):
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_ones_column(tX_grouped[i]))
    return tX_grouped_new
