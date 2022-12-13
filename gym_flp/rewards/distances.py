import math

import numpy as np

from gym_flp.util import preprocessing


def rectilinear(state1d):
    y, x = preprocessing.centroids(state1d)
    return np.array(
        [[abs(float(x[j]) - float(valx)) + abs(float(valy) - float(y[i])) for (j, valy) in enumerate(y)] for (i, valx)
         in enumerate(x)], dtype=float)


def euclidean(state1d):
    y, x = preprocessing.centroids(state1d)
    return np.array(
        [[math.sqrt(float(x[j]) - float(valx))**2 + (float(valy) - float(y[i]))**2 for (j, valy) in enumerate(y)]
         for (i, valx) in enumerate(x)], dtype=float)


def squaredEuclidean(state1d):
    y, x = preprocessing.centroids(state1d)
    return np.array(
        [[(float(x[j]) - float(valx)) ** 2 + (float(valy) - float(y[i])) ** 2 for (j, valy) in enumerate(y)]
         for (i, valx) in enumerate(x)], dtype=float)