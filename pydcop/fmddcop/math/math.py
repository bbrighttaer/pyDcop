import numpy as np


def gaussian_rbf(r):
    eps = 0.1
    return np.exp(-(eps * r)**2)

