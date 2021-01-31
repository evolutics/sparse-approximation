import math

from numpy import random
from pytest import mark
import numpy

from src.lib import randomness


class TestDrawDistribution:
    @mark.parametrize(
        "size,nonzero_count",
        [(1, 1), (2, 1), (3, 3), (8, 5), (13, None)],
    )
    def test_is_valid_distribution(self, size, nonzero_count):
        generator = random.default_rng(144)

        distribution = randomness.draw_distribution(
            generator, size, nonzero_count=nonzero_count
        )

        assert distribution.shape == (size,)
        assert all(distribution >= 0)
        assert math.isclose(numpy.sum(distribution), 1)

    def test_respects_nonzero_count(self):
        generator = random.default_rng(233)

        distribution = randomness.draw_distribution(generator, 5, nonzero_count=3)

        assert numpy.count_nonzero(distribution) == 3

    def test_randomizes_nonzero_count_if_not_given(self):
        generator = random.default_rng(377)

        nonzero_counts = [
            numpy.count_nonzero(randomness.draw_distribution(generator, 4))
            for _ in range(100)
        ]

        assert len(set(nonzero_counts)) == 4
