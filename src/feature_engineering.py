from src.proj1_helpers import *


def feature_engineering():
    raise NotImplementedError


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    a = [np.power(x, d) for d in range(2, degree+1)]
    return np.asarray(a)


def feature_expansion(tX, d):
    for feature_index in range(tX.shape[1]):
        feature = tX[:, feature_index]
        expanded_feature = build_poly(feature, d).T
        tX = np.hstack((tX,expanded_feature))   
    return tX
