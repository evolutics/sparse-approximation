import numpy

from src.lib.approximation import dense


def test_total_variation_exact_case():
    A = numpy.array([[3 / 4, 1 / 3], [1 / 4, 2 / 3]])
    b = numpy.array([3 / 5, 2 / 5])

    x = dense.total_variation(A, b)

    assert numpy.allclose(x, numpy.array([16 / 25, 9 / 25]))
