from src.proj1_helpers import *

def build_poly(features, d):
    raise NotImplementedError

def feature_expansion(tX, d):
    for feature_index in range(tX.shape[1]):
        feature = tX[:, feature_index]
        expanded_feature = build_poly(feature, d).T
        tX = np.hstack((tX,expanded_feature))   
    return tX