import math

from pytest import mark
import numpy

from src.lib import divergence
from src.lib import randomness
from src.lib.test import testing


def _cases():
    return [
        (
            divergence.euclidean,
            numpy.array([0.2, 0.3, 0.5]),
            numpy.array([0.3, 0.1, 0.6]),
            0.2449489743,
        ),
        (
            divergence.hellinger,
            numpy.array([0.2, 0.3, 0.5]),
            numpy.array([0.3, 0.1, 0.6]),
            0.1847251674,
        ),
        (
            divergence.jensen_shannon_distance,
            numpy.array([0.2, 0.3, 0.5]),
            numpy.array([0.3, 0.1, 0.6]),
            0.182953966,
        ),
        (
            divergence.k_directed,
            numpy.array([0.2, 0.3, 0.5]),
            numpy.array([0.3, 0.1, 0.6]),
            0.02935573227,
        ),
        (
            divergence.kullback_leibler,
            numpy.array([9 / 25, 12 / 25, 4 / 25]),
            numpy.array([1 / 3, 1 / 3, 1 / 3]),
            0.0852996013,
        ),
        (
            divergence.neyman_chi_square,
            numpy.array([0.2, 0.3, 0.5]),
            numpy.array([0.3, 0.1, 0.6]),
            0.45,
        ),
        (
            divergence.pearson_chi_square,
            numpy.array([0.2, 0.3, 0.5]),
            numpy.array([0.3, 0.1, 0.6]),
            0.2033333333,
        ),
        (
            divergence.total_variation,
            numpy.array([0.2, 0.3, 0.5]),
            numpy.array([0.3, 0.1, 0.6]),
            0.2,
        ),
        (divergence.total_variation, numpy.array([1, 0]), numpy.array([0, 1]), 1),
    ]


@mark.parametrize("D,p,q,expected", _cases())
def test_single(D, p, q, expected):
    actual = D(p, q)

    assert math.isclose(actual, expected)


@mark.parametrize("D,p,q,expected", _cases())
def test_broadcast(D, p, q, expected):
    Q = numpy.column_stack([p, q])

    actual = D(p, Q)

    assert actual.shape == (2,)
    assert math.isclose(actual[1], expected)


@mark.parametrize("_name,D", testing.public_functions(divergence))
def test_is_nonnegative(_name, D, generator):
    for _ in range(20):
        p = randomness.draw_distribution(generator, 5)
        q = randomness.draw_distribution(generator, 5)

        value = D(p, q)

        assert value >= 0


@mark.parametrize("_name,D", testing.public_functions(divergence))
def test_is_zero_if_same(_name, D, generator):
    for _ in range(20):
        p = randomness.draw_distribution(generator, 5)

        value = D(p, p)

        assert math.isclose(value, 0)
