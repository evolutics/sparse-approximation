import math

from numpy import linalg
from scipy import stats
from scipy.spatial import distance
import numpy


def euclidean(p, Q):
    return _column_wise(Q, lambda q: linalg.norm(p - q))


def hellinger(p, Q):
    factor = 1 / math.sqrt(2)
    sqrt_p = numpy.sqrt(p)
    return factor * _column_wise(Q, lambda q: linalg.norm(sqrt_p - numpy.sqrt(q)))


def jensen_shannon_distance(p, Q):
    """Square root of Jensen-Shannon divergence."""

    return _column_wise(Q, lambda q: distance.jensenshannon(p, q))


def k_directed(p, Q):
    """See: Jianhua Lin. "Divergence Measures Based on the Shannon Entropy". 1991."""

    return _column_wise(Q, lambda q: stats.entropy(p, (p + q) / 2))


def kullback_leibler(p, Q):
    return _column_wise(Q, lambda q: stats.entropy(p, q))


def neyman_chi_square(p, Q):
    return _column_wise(Q, lambda q: numpy.sum(numpy.square(p - q) / q))


def pearson_chi_square(p, Q):
    return _column_wise(Q, lambda q: numpy.sum(numpy.square(p - q) / p))


def total_variation(p, Q):
    return 0.5 * _column_wise(Q, lambda q: linalg.norm(p - q, 1))


def _column_wise(matrix, function):
    return numpy.apply_along_axis(function, axis=0, arr=matrix)
