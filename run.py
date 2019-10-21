import numpy as np

import preprocessing as prep
import feature_engineering as f_e
import local_prediction as pred
import proj1_helpers as helpers
import params

if __name__ == '__main__':
    # Training set preprocessing and feature engineering
    print('Train set:')
    y, tX, ids = helpers.load_csv_data(params.DATA_TRAIN_PATH)
    y_preprocessed, tX_preprocessed, ids_preprocessed, masks, counts = prep.preprocess(y, tX, ids)
    tX_improved = f_e.feature_engineer(tX_preprocessed)

    # In case we want to test our model locally by splitting our data
    if params.LOCAL_PREDICTION:
        pred.locally_predict(tX_improved, y_preprocessed, counts)
    else:
        print('Test set:')
        y_test, tX_test, ids_test = helpers.load_csv_data(params.DATA_TEST_PATH)
        y_test_preprocessed, tX_test_preprocessed, ids_test_preprocessed, masks_test, counts_test = prep.preprocess(
            y_test,
            tX_test,
            ids_test)
        tX_test_improved = f_e.feature_engineer(tX_test_preprocessed)
        log_initial_ws = []
        for i in range(len(tX_test_improved)):
            log_initial_ws.append(np.repeat(0, tX_test_improved[i].shape[1]))
        optimal_ws = pred.find_optimal_ws_grouped(tX_test_improved, y_test_preprocessed, params.IMPLEMENTATION,
                                                  log_initial_ws,
                                                  params.MAX_ITERS, params.GAMMA, params.REGULATOR, params.RIDGE_LAMBDA)
        y_preds = []
        for i in range(len(optimal_ws)):
            y_preds.append(helpers.predict_labels(optimal_ws[i], tX_test_improved[i], params.IMPLEMENTATION)[1])
        flat_y_preds = helpers.flatten_list(y_preds)
        flat_ids = helpers.flatten_list(ids_test_preprocessed)
        ids_indices = np.argsort(flat_ids)
        y_preds_sorted = np.array(flat_y_preds)[ids_indices]
        helpers.create_csv_submission(ids_test, y_preds_sorted, params.OUTPUT_PATH)
