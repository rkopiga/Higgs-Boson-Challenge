import numpy as np

def load_data(path_dataset):
    """Load data and convert it to the metrics system."""
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=list(range(32)))
    return data