import math

from numpy import linalg
from scipy import stats
from scipy.spatial import distance
import numpy


def hellinger_distance(p, Q):
    sqrt_p = numpy.sqrt(p)
    return (1 / math.sqrt(2)) * _column_wise(
        Q, lambda q: linalg.norm(sqrt_p - numpy.sqrt(q))
    )


def jensen_shannon_distance(p, Q):
    return _column_wise(Q, lambda q: distance.jensenshannon(p, q))


def k_directed_divergence(p, Q):
    """See: Jianhua Lin. "Divergence Measures Based on the Shannon Entropy". 1991."""

    return _column_wise(Q, lambda q: stats.entropy(p, (p + q) / 2))


def kullback_leibler(p, Q):
    return _column_wise(Q, lambda q: stats.entropy(p, q))


def squared_euclidean(p, Q):
    return numpy.square(_column_wise(Q, lambda q: linalg.norm(p - q)))


def total_variation(p, Q):
    return 0.5 * _column_wise(Q, lambda q: linalg.norm(p - q, 1))


def _column_wise(matrix, function):
    return numpy.apply_along_axis(function, axis=0, arr=matrix)
