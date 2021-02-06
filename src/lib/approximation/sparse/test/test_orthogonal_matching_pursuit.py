import numpy

from src.lib import divergence
from src.lib import normalize
from src.lib.approximation import dense
from src.lib.approximation.sparse import orthogonal_matching_pursuit


def test_selects_single_atom_for_total_variation():
    A = numpy.array([[1, 0.5, 0.4, 0.8], [0, 0.5, 0.6, 0.2]])
    b = numpy.array([0, 1])
    K = 1

    x = orthogonal_matching_pursuit.solve(
        A,
        b,
        K,
        solve_dense=dense.total_variation,
        potential=lambda r, A: divergence.total_variation(normalize.clip(r), A),
    )

    assert numpy.allclose(x, numpy.array([0, 0, 1, 0]))
