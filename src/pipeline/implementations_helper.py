from src.pipeline.implementations import *


def find_optimal_ws_LS_grouped(tX_grouped, y_grouped):
    optimal_ws = []
    for i in range(len(tX_grouped)):
        optimal_w, _ = least_squares(y_grouped[i], tX_grouped[i])
        optimal_ws.append(optimal_w)
    return optimal_ws


def find_optimal_ws_Ridge_grouped(tX_grouped, y_grouped, initial_w_grouped, max_iters_grouped, gamma_grouped,
                                  regulator_grouped):
    optimal_ws = []
    for i in range(len(tX_grouped)):
        optimal_w, _ = logistic_regression_GD(y_grouped[i], tX_grouped, initial_w_grouped[i], max_iters_grouped[i],
                                              gamma_grouped[i], regulator_grouped[i])
        optimal_ws.append(optimal_w)
    return optimal_ws
