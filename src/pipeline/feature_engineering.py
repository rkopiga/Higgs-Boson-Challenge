from src.pipeline.proj1_helpers import *
from src.params import *


def feature_engineering(tX,
                        group=GROUP,
                        polynomial_expansion=FEATURE_EXPANSION,
                        degree=DEGREE):
    if group:
        if polynomial_expansion:
            tX = feature_expansion_grouped(tX, degree)
    else:
        if polynomial_expansion:
            tX = feature_expansion(tX, degree)
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
