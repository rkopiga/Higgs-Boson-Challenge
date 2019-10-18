# Global variables
DATA_TRAIN_PATH = '../../data/train.csv'
DATA_TEST_PATH = '../../data/test.csv'
OUTPUT_PATH = '../../data/output.csv'
DEBUG = False

# /!\ DO NOT CHANGE VALUES BELOW /!\
UNWANTED_VALUE = -999
PRI_jet_num_max_value = 3
PRI_jet_num_index = 22
# /!\ -------------------------- /!\

# Preprocessing parameters
SHUFFLE_DATA = True
GROUP_1 = False
GROUP_2 = False
GROUP = GROUP_1 or GROUP_2
ADDITIONAL_SPLITTING = True
REPLACE_UNWANTED_VALUE = False
REMOVE_INV_FEATURES = True
STD = True

# Feature engineering parameters
FEATURE_EXPANSION = False
DEGREE = 2
MULTIPLICATION_COMBINATIONS = 3
ONE_COLUMN = False

# Local prediction parameters
LOCAL_PREDICTION = True
RATIO = 8/10

# Implementations parameters
LOGISTIC_MODEL = True
MAX_ITERS = 25
GAMMA = 0.05
REGULATOR = 1
