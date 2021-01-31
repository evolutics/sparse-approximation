from numpy import linalg
import numpy


def total_variation(p, Q):
    return 0.5 * _column_wise(Q, lambda q: linalg.norm(p - q, 1))


def _column_wise(matrix, function):
    return numpy.apply_along_axis(function, axis=0, arr=matrix)
