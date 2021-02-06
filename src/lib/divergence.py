import math

from numpy import linalg
from scipy import stats
from scipy.spatial import distance
import numpy


def euclidean(p, Q):
    return numpy.apply_along_axis(lambda q: linalg.norm(p - q), 0, Q)


def hellinger(p, Q):
    factor = 1 / math.sqrt(2)
    sqrt_p = numpy.sqrt(p)
    return factor * numpy.apply_along_axis(
        lambda q: linalg.norm(sqrt_p - numpy.sqrt(q)), 0, Q
    )


def jensen_shannon_distance(p, Q):
    """Square root of Jensen-Shannon divergence."""

    return numpy.apply_along_axis(lambda q: distance.jensenshannon(p, q), 0, Q)


def k_directed(p, Q):
    """See: Jianhua Lin. "Divergence Measures Based on the Shannon Entropy". 1991."""

    return numpy.apply_along_axis(lambda q: stats.entropy(p, (p + q) / 2), 0, Q)


def kullback_leibler(p, Q):
    return numpy.apply_along_axis(lambda q: stats.entropy(p, q), 0, Q)


def neyman_chi_square(p, Q):
    return numpy.apply_along_axis(lambda q: numpy.sum(numpy.square(p - q) / q), 0, Q)


def pearson_chi_square(p, Q):
    return numpy.apply_along_axis(lambda q: numpy.sum(numpy.square(p - q) / p), 0, Q)


def total_variation(p, Q):
    return 0.5 * numpy.apply_along_axis(lambda q: linalg.norm(p - q, 1), 0, Q)
