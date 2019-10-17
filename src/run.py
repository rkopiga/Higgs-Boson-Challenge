from src.pipeline.preprocessing import *
from src.pipeline.feature_engineering import *
from src.pipeline.local_prediction import *
from src.pipeline.helpers import *

# Training set preprocessing and feature engineering
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
y_preprocessed, tX_preprocessed, ids_preprocessed, masks, counts = preprocess(y, tX, ids)
tX_improved = feature_engineering(tX_preprocessed)

# In case we want to test our model locally by splitting our data
if LOCAL_PREDICTION:
    local_prediction(tX_improved, y_preprocessed)
else:
    # In case we want to use our model on the actual test set
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    y_test_preprocessed, tX_test_preprocessed, ids_test_preprocessed, masks_test, counts_test = preprocess(y_test, tX_test, ids_test)
    tX_test_improved = feature_engineering(tX_test_preprocessed)
    optimal_ws = find_optimal_ws_LS_grouped(tX_improved, y_preprocessed)
    y_preds = []
    for i in range(len(optimal_ws)):
        y_preds.append(predict_labels(optimal_ws[i], tX_test_improved[i]))
    flat_y_preds = flatten_list(y_preds)
    flat_ids = flatten_list(ids_test_preprocessed)
    ids_indices = np.argsort(flat_ids)
    y_preds_sorted = np.array(flat_y_preds)[ids_indices]
