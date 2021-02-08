from pytest import mark
import numpy

from src.lib import divergence
from src.lib import normalize
from src.lib.approximation import dense
from src.lib.approximation.sparse import brute_force_search
from src.lib.approximation.sparse import frank_wolfe
from src.lib.approximation.sparse import orthogonal_matching_pursuit


def _cases():
    return [
        lambda *problem: brute_force_search.solve(
            *problem,
            D=divergence.total_variation,
            solve_dense=dense.total_variation,
        ),
        lambda *problem: frank_wolfe.solve(
            *problem,
            solve_dense=dense.total_variation,
            potential=lambda r, A: divergence.total_variation(normalize.clip(r), A),
            is_step_size_adaptive=False,
        ),
        lambda *problem: frank_wolfe.solve(
            *problem,
            solve_dense=dense.total_variation,
            potential=lambda r, A: divergence.total_variation(normalize.clip(r), A),
            is_step_size_adaptive=True,
        ),
        lambda *problem: orthogonal_matching_pursuit.solve(
            *problem,
            solve_dense=dense.euclidean,
            potential=lambda r, A: -A.T @ r,
        ),
        lambda *problem: orthogonal_matching_pursuit.solve(
            *problem,
            solve_dense=dense.total_variation,
            potential=lambda r, A: divergence.total_variation(normalize.clip(r), A),
        ),
    ]


@mark.parametrize("solve", _cases())
def test_selects_single_atom(solve):
    A = numpy.array([[1, 0.5, 0.4, 0.8], [0, 0.5, 0.6, 0.2]])
    b = numpy.array([0, 1])
    K = 1

    x = solve(A, b, K)

    assert numpy.allclose(x, numpy.array([0, 0, 1, 0]))