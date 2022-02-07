from pytest import mark
import numpy

from src.lib.approximation import dense
from src.lib.test import testing


@mark.parametrize("_name,solve", testing.public_functions(dense))
def test_exact_case(_name, solve):
    C = numpy.array([[3 / 4, 1 / 3], [1 / 4, 2 / 3]])
    p = numpy.array([3 / 5, 2 / 5])

    y = solve(C, p)

    assert numpy.allclose(y, numpy.array([16 / 25, 9 / 25]))
