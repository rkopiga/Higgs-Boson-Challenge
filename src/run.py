from src.pipeline.preprocessing import *
from src.pipeline.feature_engineering import *
from src.params import *

# Training set
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
y_preprocessed, tX_preprocessed, ids_preprocessed, masks, counts = preprocess(y, tX, ids)
tX_improved = feature_engineering(tX_preprocessed)

# Test set
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
y_test_preprocessed, tX_test_preprocessed, ids_test_preprocessed, masks_test, counts_test = preprocess(y, tX, ids)
tX_test_improved = feature_engineering(tX_test_preprocessed)