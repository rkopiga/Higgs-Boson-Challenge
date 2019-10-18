from src.params import *
from src.pipeline.implementations_helper import *
from src.pipeline.proj1_helpers import *
import matplotlib.pyplot as plt


def local_prediction(tX, y, group=GROUP, ratio=RATIO):
    if group:
        tX_train_grouped, y_train_grouped, tX_sub_test_grouped, y_sub_test_grouped = separate_data_grouped(tX, y, ratio)
        optimal_ws = find_optimal_ws_LS_grouped(tX_train_grouped, y_train_grouped)
        y_pred_grouped = []
        y_pred_clipped_grouped = []
        for i in range(len(optimal_ws)):
            y_pred, y_pred_clipped = predict_labels(optimal_ws[i], tX_sub_test_grouped[i])
            y_pred_grouped.append(y_pred)
            y_pred_clipped_grouped.append(y_pred_clipped)
        compare_labels_grouped(y_pred_grouped, y_pred_clipped_grouped, y_sub_test_grouped)
    else:
        tX_train, y_train, tX_sub_test, y_sub_test = separate_data(tX, y, ratio)
        optimal_w, _ = least_squares(tX_train, y_train)
        y_pred, y_pred_clipped = predict_labels(optimal_w, tX_sub_test)
        compare_labels(y_pred, y_pred_clipped, y_sub_test)


def separate_data(tX, y, ratio):
    index = int(len(y) * ratio)
    tX_train = tX[:index]
    y_train = y[:index]
    tX_sub_test = tX[index:]
    y_sub_test = y[index:]
    return tX_train, y_train, tX_sub_test, y_sub_test


def separate_data_grouped(tX_grouped, y_grouped, ratio):
    tX_train_grouped = []
    y_train_grouped = []
    tX_sub_test_grouped = []
    y_sub_test_grouped = []
    for i in range(len(tX_grouped)):
        tX_train, y_train, tX_sub_test, y_sub_test = separate_data(tX_grouped[i], y_grouped[i], ratio)
        tX_train_grouped.append(tX_train)
        y_train_grouped.append(y_train)
        tX_sub_test_grouped.append(tX_sub_test)
        y_sub_test_grouped.append(y_sub_test)
    return tX_train_grouped, y_train_grouped, tX_sub_test_grouped, y_sub_test_grouped


def compare_labels(y_pred, y_pred_clipped, y_sub_test):
    accuracy = np.abs(y_pred_clipped + y_sub_test)
    unique, counts = np.unique(accuracy, return_counts=True)
    print("\tAccuracy : {}".format(counts[1] / len(accuracy)))
    plt.figure()
    plt.hist(y_pred, bins='auto')


def compare_labels_grouped(y_pred_grouped, y_pred_clipped_grouped, y_sub_test_grouped):
    for i in range(len(y_pred_clipped_grouped)):
        print("Group {}:".format(i))
        compare_labels(y_pred_grouped[i], y_pred_clipped_grouped[i], y_sub_test_grouped[i])
