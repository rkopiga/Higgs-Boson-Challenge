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
GROUP_1 = False
GROUP_2 = True
GROUP_2_ADDITIONAL_SPLITTING = True
GROUP = GROUP_1 or GROUP_2
REMOVE_INV_FEATURES = True
REPLACE_UNWANTED_VALUE = True
STD = True
REPLACE_OUTLIERS = True
THRESHOLD = 1.5

# Feature engineering parameters ---------------------------------------------------------------------------------------
FEATURE_EXPANSION = False
DEGREE = 3
FEATURE_MULTIPLICATION = True
ADD_COS = False
ADD_SIN = True
ADD_TAN = False  # Not good
ADD_EXP = False
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
MAX_ITERS = 50
GAMMA = 0.05
r = 1  # Between 0.5 and 1
LOG_LAMBDA = 0

# Ridge-regression parameters
RIDGE_LAMBDA = 0.000000001
