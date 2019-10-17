from src.proj1_helpers import *
from src.params import *


def feature_engineering(tX, polynomial_expansion=FEATURE_EXPANSION):
    if GROUP_1 or GROUP_2:
        if polynomial_expansion:
            tX = feature_expansion_grouped(tX)
    else:
        if polynomial_expansion:
            tX = feature_expansion(tX)
    return tX


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    a = [np.power(x, d) for d in range(2, degree+1)]
    return np.asarray(a)


def feature_expansion(tX, d=DEGREE):
    for feature_index in range(tX.shape[1]):
        feature = tX[:, feature_index]
        expanded_feature = build_poly(feature, d).T
        tX = np.hstack((tX, expanded_feature))
    return tX


def feature_expansion_grouped(tX_grouped):
    tX_expanded = []
    for i in range(len(tX_grouped)):
        tX_expanded.append(feature_expansion(tX_grouped[i]))
    return tX_expanded
