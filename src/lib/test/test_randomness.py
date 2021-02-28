import math

from pytest import mark
import numpy

from src.lib import randomness


class TestDrawDistribution:
    @mark.parametrize(
        "size,nonzero_range",
        [
            (1, (1, 2)),
            (2, (1, 3)),
            (3, (3, 4)),
            (8, (3, 6)),
            (13, None),
        ],
    )
    def test_is_valid_distribution(self, size, nonzero_range, generator):
        distribution = randomness.draw_distribution(
            generator, size, nonzero_range=nonzero_range
        )

        assert distribution.shape == (size,)
        assert all(distribution >= 0)
        assert math.isclose(numpy.sum(distribution), 1)

    @mark.parametrize(
        "size,nonzero_range,expected_nonzero_counts",
        [
            (1, (1, 2), {1}),
            (4, (1, 2), {1}),
            (4, (1, 5), {1, 2, 3, 4}),
            (4, (3, 4), {3}),
            (4, (3, 5), {3, 4}),
            (4, None, {4}),
        ],
    )
    def test_randomizes_nonzero_count_within_range(
        self, size, nonzero_range, expected_nonzero_counts, generator
    ):
        actual_nonzero_counts = {
            numpy.count_nonzero(
                randomness.draw_distribution(
                    generator, size, nonzero_range=nonzero_range
                )
            )
            for _ in range(100)
        }

        assert actual_nonzero_counts == expected_nonzero_counts


class TestDrawDistributions:
    def test_are_valid_distributions(self, generator):
        distributions = randomness.draw_distributions(generator, 5, 8)

        assert distributions.shape == (5, 8)
        assert numpy.all(distributions >= 0)
        assert numpy.allclose(numpy.sum(distributions, axis=0), 1)
