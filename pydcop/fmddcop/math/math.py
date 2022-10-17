import numpy as np
import math


def gaussian_rbf(r):
    eps = 0.1
    return np.exp(-(eps * r)**2)


def euclidean_distance(point1, point2):
    """
    Returns the Euclidean distance between two points on a plane.
    """
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
