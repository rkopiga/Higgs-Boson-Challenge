# Global variables
DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'
OUTPUT_PATH = 'data/output.csv'
DEBUG = True

# /!\ DO NOT CHANGE VALUES BELOW /!\
UNWANTED_VALUE = -999
PRI_jet_num_max_value = 3
PRI_jet_num_index = 22
# /!\ -------------------------- /!\

# Preprocessing parameters ---------------------------------------------------------------------------------------------
SHUFFLE_DATA = False
GROUP_1 = False
GROUP_2 = True
GROUP_2_ADDITIONAL_SPLITTING = True
GROUP = GROUP_1 or GROUP_2
REMOVE_INV_FEATURES = True
REPLACE_UNWANTED_VALUE = True
STD = True

# Feature engineering parameters ---------------------------------------------------------------------------------------
FEATURE_EXPANSION = False
DEGREE = 2
FEATURE_MULTIPLICATION = True
ONE_COLUMN = True

# Local prediction parameters ------------------------------------------------------------------------------------------
LOCAL_PREDICTION = True
RATIO = 8/10

# Implementations parameters -------------------------------------------------------------------------------------------
# 0 = Least-squares
# 1 = Ridge-regression
# 2 = Logistic-regression
IMPLEMENTATION = 1

# Logistic-regression parameters
MAX_ITERS = 25
GAMMA = 0.05
REGULATOR = 1

# Ridge-regression parameters
RIDGE_LAMBDA = 0.000001
