import preprocessing as prep
import local_prediction as pred
import feature_engineering as f_e
import params
import proj1_helpers as helpers

implementations_linspace = [0, 0, 0]

for i in range(len(implementations_linspace)):
    implementations_linspace[i] = 1

y, tX, ids = helpers.load_csv_data(params.DATA_TRAIN_PATH)
y_preprocessed, tX_preprocessed, ids_preprocessed, masks, counts = prep.preprocess(y, tX, ids)
tX_improved = f_e.feature_engineer(tX_preprocessed)
accuracy = pred.locally_predict(tX_improved, y_preprocessed, params.IMPLEMENTATION)
