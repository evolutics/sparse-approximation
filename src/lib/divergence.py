from numpy import linalg
from scipy import stats
import numpy


def kullback_leibler(p, Q):
    return _column_wise(Q, lambda q: stats.entropy(p, q))


def squared_euclidean(p, Q):
    return numpy.square(_column_wise(Q, lambda q: linalg.norm(p - q)))


def total_variation(p, Q):
    return 0.5 * _column_wise(Q, lambda q: linalg.norm(p - q, 1))


def _column_wise(matrix, function):
    return numpy.apply_along_axis(function, axis=0, arr=matrix)
