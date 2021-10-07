import fnmatch
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def get_synfiles(path):
    """
    Returns a list of absolute paths to all .syn files in a given directory
    """
    filenames = []
    for dirpath, dirs, files in os.walk(path):
        for nm_f in fnmatch.filter(files, "*.mat"):
            filenames.append(os.path.join(path, nm_f))

    num_files = len(filenames)
    print("Loaded {} synfiles from {}".format(num_files, path))
    return filenames


def load_synfile(filepath):
    """
    Load a .syn file and extract scoring data
    """
    data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)[
        "scoringData"
    ]
    return data


def load_ephys_data(filepath):
    """
    Load MATLAB file containing electrophysiology data from uncaging experiments
    """

    matfile = scipy.io.loadmat(
        filepath, struct_as_record=False, squeeze_me=True, chars_as_strings=True
    )

    for k,v in matfile.items():
        if 'puncta' in k:
            data = v

    return data


