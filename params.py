# Global variables
DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'
OUTPUT_PATH = 'data/output.csv'
DEBUG = True

# /!\ DO NOT CHANGE VALUES BELOW /!\
UNWANTED_VALUE = -999
PRI_jet_num_max_value = 3
PRI_jet_num_index = 22
PHIs_indices = [15, 18, 20, 25, 28]
PRI_jet_num_new_index = PRI_jet_num_index - 3
# /!\ -------------------------- /!\

# Preprocessing parameters ---------------------------------------------------------------------------------------------
SHUFFLE_DATA = True
REMOVE_PHIS = True
ADD_DIFF_PHIS = False  # TODO
GROUP_1 = False
GROUP_2 = True
LESS_GROUPS = True
GROUPS_TO_MERGE = [2, 3]
GROUP_2_ADDITIONAL_SPLITTING = True
GROUP = GROUP_1 or GROUP_2
REMOVE_INV_FEATURES = True
REPLACE_UNWANTED_VALUE = True
VALUE = 'median'  # possible values: median, mean
STD = True
REPLACE_OUTLIERS = True
OUTLIER_VALUE = 'clip'  # possible values: clip, mean, upper_lower_mean
THRESHOLD = 1.5
REMOVE_DUPLICATE_FEATURES = True

# Feature engineering parameters ---------------------------------------------------------------------------------------
FEATURE_EXPANSION = False
DEGREE = 3
FEATURE_MULTIPLICATION = True
ADD_COS = False
ADD_SIN = True
ADD_TAN = False  # Not good
ADD_EXP = False
ADD_LOG = False
ADD_SQRT = False
ADD_COS2 = False
ADD_SIN2 = False
ONE_COLUMN = True

# Local prediction parameters ------------------------------------------------------------------------------------------
LOCAL_PREDICTION = True
RATIO = 8/10
CROSS_VALIDATION = True
K = 3

# Implementations parameters -------------------------------------------------------------------------------------------
# 0 = Least-squares
# 1 = Ridge-regression
# 2 = Logistic-regression
IMPLEMENTATION = 2

# Logistic-regression parameters
MAX_ITERS = 30
GAMMA = 0.05
DECREASING_GAMMA = True
r = 0.5  # Between 0.5 and 1
LOG_LAMBDA = 0.000000001

# Ridge-regression parameters
# RIDGE_LAMBDA = 0.000000001
# RIDGE_LAMBDA = 0.0001
# RIDGE_LAMBDA = 0.0000001
# RIDGE_LAMBDA = 0.00000001
RIDGE_LAMBDA = 0.00000001


# TODO add arguments to the function to be able to put it at the beginning of locally_predict
def print_parameters():
    print('----- Preprocessing parameters -----\n')
    print('REMOVE_PHIS = {}'.format(REMOVE_PHIS))
    print('ADD_DIFF_PHIS = {}'.format(ADD_DIFF_PHIS))
    print('GROUP_1 = {}'.format(GROUP_1))
    print('GROUP_2 = {}'.format(GROUP_2))
    if GROUP_2:
        print('\tGROUP_2_ADDITIONAL_SPLITTING = {}'.format(GROUP_2_ADDITIONAL_SPLITTING))
    print('REMOVE_INV_FEATURES = {}'.format(REMOVE_INV_FEATURES))
    print('REPLACE_UNWANTED_VALUE = {}'.format(REPLACE_UNWANTED_VALUE))
    print('STD = {}'.format(STD))
    print('REPLACE_OUTLIERS = {}'.format(REPLACE_OUTLIERS))
    if REPLACE_OUTLIERS:
        print('\tTHRESHOLD = {}'.format(THRESHOLD))

    print('----- Feature engineering parameters -----\n')
    print('FEATURE_EXPANSION = {}'.format(FEATURE_EXPANSION))
    if FEATURE_EXPANSION:
        print('DEGREE = {}'.format(DEGREE))
    print('FEATURE_MULTIPLICATION = {}'.format(FEATURE_MULTIPLICATION))
    print('ADD_COS = {}'.format(ADD_COS))
    print('ADD_SIN = {}'.format(ADD_SIN))
    print('ADD_TAN = {}'.format(ADD_TAN))
    print('ADD_EXP = {}'.format(ADD_EXP))
    print('ADD_LOG = {}'.format(ADD_LOG))
    print('ADD_SQRT = {}'.format(ADD_SQRT))
    print('ADD_COS2 = {}'.format(ADD_COS2))
    print('ADD_SIN2 = {}'.format(ADD_SIN2))
    print('ONE_COLUMN = {}'.format(ONE_COLUMN))

    if LOCAL_PREDICTION:
        print('----- Local prediction parameters -----\n')
        print('RATIO = {}'.format(RATIO))
        print('CROSS_VALIDATION = {}'.format(CROSS_VALIDATION))

    print('----- Implementation parameters -----\n')
    print('IMPLEMENTATION = {}'.format(IMPLEMENTATION))
    if IMPLEMENTATION == 2:
        print('MAX_ITERS = {}'.format(MAX_ITERS))
        print('GAMMA = {}'.format(GAMMA))
        print('r = {}'.format(r))
        print('LOG_LAMBDA = {}'.format(LOG_LAMBDA))
    if IMPLEMENTATION == 1:
        print('RIDGE_LAMBDA = {}'.format(RIDGE_LAMBDA))
