from preprocessing import *
from feature_engineering import *
from local_prediction import *
from helpers import *

# Training set preprocessing and feature engineering
print('Train set:')
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
y_preprocessed, tX_preprocessed, ids_preprocessed, masks, counts = preprocess(y, tX, ids)
tX_improved = feature_engineering(tX_preprocessed)

# In case we want to test our model locally by splitting our data
if LOCAL_PREDICTION:
    local_prediction(tX_improved, y_preprocessed, counts)
else:
    print('Test set:')
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    y_test_preprocessed, tX_test_preprocessed, ids_test_preprocessed, masks_test, counts_test = preprocess(y_test,
                                                                                                           tX_test,
                                                                                                           ids_test)
    tX_test_improved = feature_engineering(tX_test_preprocessed)
    log_initial_ws = []
    for i in range(len(tX_test_improved)):
        log_initial_ws.append(np.repeat(0, tX_test_improved[i].shape[1]))
    optimal_ws = find_optimal_ws_grouped(tX_test_improved, y_test_preprocessed, IMPLEMENTATION, log_initial_ws,
                                         MAX_ITERS, GAMMA, REGULATOR, RIDGE_LAMBDA)
    y_preds = []
    for i in range(len(optimal_ws)):
        y_preds.append(predict_labels(optimal_ws[i], tX_test_improved[i], IMPLEMENTATION)[1])
    flat_y_preds = flatten_list(y_preds)
    flat_ids = flatten_list(ids_test_preprocessed)
    ids_indices = np.argsort(flat_ids)
    y_preds_sorted = np.array(flat_y_preds)[ids_indices]
    create_csv_submission(ids_test, y_preds_sorted, OUTPUT_PATH)
