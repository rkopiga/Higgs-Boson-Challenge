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
SHUFFLE_DATA = True
GROUP_1 = False
GROUP_2 = True
GROUP = GROUP_1 or GROUP_2
ADDITIONAL_SPLITTING = True
REPLACE_UNWANTED_VALUE = True
REMOVE_INV_FEATURES = True
STD = False

# Feature engineering parameters ---------------------------------------------------------------------------------------
FEATURE_EXPANSION = True
DEGREE = 2
FEATURE_MULTIPLICATION = False
ONE_COLUMN = False

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
RIDGE_LAMBDA = 0.00001
