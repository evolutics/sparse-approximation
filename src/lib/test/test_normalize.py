import numpy

from src.lib import normalize


class TestClip:
    def test_keeps_valid_distribution(self):
        assert numpy.allclose(
            normalize.clip(numpy.array([0.4, 0.6])), numpy.array([0.4, 0.6])
        )

    def test_clips(self):
        assert numpy.allclose(
            normalize.clip(numpy.array([-0.2, 0.4, 1.2])), numpy.array([0, 0.25, 0.75])
        )
