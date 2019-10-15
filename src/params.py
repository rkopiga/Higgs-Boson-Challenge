# Used to group the data according to that value and to replace it by an appropriate value
UNWANTED_VALUE = -999

# Used to group the data according to the PRI_jet_num feature
PRI_jet_num_max_value = 3
PRI_jet_num_index = 22  # when starting at 0

# Preprocessing parameters
GROUP_1 = False
GROUP_2 = True
REPLACE_UNWANTED_VALUE=True
REMOVE_INV_FEATURES=True
STD=False

# Feature engineering parameters
DEGREES = [2, 3, 7]

# Cross validation parameters
SUB_TEST_SETS = True
RATIO = 3/4
