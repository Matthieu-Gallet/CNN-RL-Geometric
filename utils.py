import numpy as np
import pickle
from os.path import dirname, join, makedirs
import time, os, pickle
from tqdm import tqdm


def dump_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return 1


def open_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def norm_one(X_train):
    return (X_train - X_train.min()) / (X_train.max() - X_train.min())


def normalize(X_train, a, b):
    """Normalize the data between a and b

    Parameters
    ----------
    X_train : numpy array
        Input data
    a : float
        Lower bound
    b : float
        Upper bound

    Returns
    -------
    numpy array
        Normalized data
    """
    X_train = (a - b) * norm_one(X_train) + b
    return X_train
