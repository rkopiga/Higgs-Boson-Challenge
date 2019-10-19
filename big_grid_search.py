from preprocessing import *
from local_prediction import *
from feature_engineering import *

implemetations_linspace = [0, 0, 0]

for i in range(len(implemetations_linspace)):
    implemetations_linspace[i] = 1

y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
y_preprocessed, tX_preprocessed, ids_preprocessed, masks, counts = preprocess(y, tX, ids)
tX_improved = feature_engineering(tX_preprocessed)
local_prediction(tX_improved, y_preprocessed, IMPLEMENTATION)
