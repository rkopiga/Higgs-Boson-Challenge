from src.implementations import *

def find_optimal_ws_LS_in_groups(tX_preprocessed, y_preprocessed):
    optimal_ws = []
    for i in range(len(tX_preprocessed)):
        optimal_w, _ = least_squares(y_preprocessed[i], tX_preprocessed[i])
        optimal_ws.append(optimal_w)
    return optimal_ws
