# Global variables
DATA_TRAIN_PATH = '../../data/train.csv'
DATA_TEST_PATH = '../../data/test.csv'
OUTPUT_PATH = '../../data/output.csv'

# /!\ DO NOT CHANGE VALUES BELOW /!\
UNWANTED_VALUE = -999
PRI_jet_num_max_value = 3
PRI_jet_num_index = 22
# /!\ -------------------------- /!\

# Preprocessing parameters
SHUFFLE_DATA = True
GROUP_1 = True
GROUP_2 = False
GROUP = GROUP_1 or GROUP_2
REPLACE_UNWANTED_VALUE = True
REMOVE_INV_FEATURES = True
STD = False

# Feature engineering parameters
FEATURE_EXPANSION = True
DEGREE = 2
MULTIPLICATION_COMBINATIONS = 3

# Local prediction parameters
LOCAL_PREDICTION = True
RATIO = 8/10
