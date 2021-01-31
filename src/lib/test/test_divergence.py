import math

from numpy import random
from pytest import mark
import numpy

from src.lib import divergence
from src.lib import randomness


def _cases():
    return [
        (
            divergence.total_variation,
            numpy.array([0.2, 0.3, 0.5]),
            numpy.array([0.1, 0.2, 0.7]),
            0.2,
        )
    ]


@mark.parametrize(
    "D,p,q,expected",
    _cases(),
)
def test_single(D, p, q, expected):
    actual = D(p, q)

    assert math.isclose(actual, expected)


@mark.parametrize(
    "D,p,q,expected",
    _cases(),
)
def test_broadcast(D, p, q, expected):
    Q = numpy.column_stack([p, q])

    actual = D(p, Q)

    assert actual.shape == (2,)
    assert math.isclose(actual[1], expected)


@mark.parametrize(
    "D",
    [case[0] for case in _cases()],
)
def test_is_nonnegative(D):
    generator = random.default_rng(144)
    for _ in range(20):
        p = randomness.draw_distribution(generator, 5)
        q = randomness.draw_distribution(generator, 5)

        value = D(p, q)

        assert value >= 0


@mark.parametrize(
    "D",
    [case[0] for case in _cases()],
)
def test_is_zero_if_same(D):
    generator = random.default_rng(233)
    for _ in range(20):
        p = randomness.draw_distribution(generator, 5)

        value = D(p, p)

        assert math.isclose(value, 0)
