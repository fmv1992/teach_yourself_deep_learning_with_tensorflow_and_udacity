"""Common library for this course."""

import matplotlib.pyplot as plt
import numpy as np
import os


def get_root_dir_based_on_dotgit(path):
    """Scan the folder iteratively for the '.git' root folder."""
    if os.path.isfile(path):
        _this_file = os.path.abspath(__file__)
        _this_folder = os.path.dirname(_this_file)
    else:
        _this_folder = os.path.abspath(path)
    while '.git' not in os.listdir(_this_folder):
        _this_folder = os.path.dirname(_this_folder)
    return os.path.abspath(_this_folder)


def plot_timed_sample_from_array(arr, n=10, duration_ms=1000):
    """Plot n images from array of shape (a, b, c) with shape (b, c)."""
    for _ in range(n):
        grid = arr[np.random.randint(0, arr.shape[0]), :]
        fig = plt.figure()
        timer = fig.canvas.new_timer(interval=1000)
        timer.add_callback(plt.close)
        img = plt.imshow(grid, cmap='gray', vmin=grid.min(), vmax=grid.max())
        timer.start()
        plt.show()
        plt.close('all')


# Define program constants.
train_size = 10000
validation_size = 1000
test_size = 1000

# Define path constants.
ROOT_PATH = get_root_dir_based_on_dotgit(__file__)
DATA_PATH = os.path.join(ROOT_PATH, 'data')

assert os.path.exists(ROOT_PATH)
assert os.path.exists(DATA_PATH)

DATASET_PICKLE_FILE = os.path.join(DATA_PATH, 'notMNIST.pickle')
