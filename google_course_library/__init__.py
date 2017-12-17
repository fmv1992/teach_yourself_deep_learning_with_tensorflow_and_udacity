"""Common library for this course."""

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


ROOT_PATH = get_root_dir_based_on_dotgit()
DATA_PATH = os.path.join(ROOT_PATH, 'data')

assert os.path.exists(ROOT_PATH)
assert os.path.exists(DATA_PATH)
