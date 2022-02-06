from pytest import mark
import numpy

from src.lib import divergence
from src.lib import normalize
from src.lib.approximation import dense
from src.lib.approximation.sparse import brute_force_search
from src.lib.approximation.sparse import compressive_sampling_matching_pursuit
from src.lib.approximation.sparse import forward_backward_pursuit
from src.lib.approximation.sparse import frank_wolfe
from src.lib.approximation.sparse import generalized_orthogonal_matching_pursuit
from src.lib.approximation.sparse import generalized_reverse_matching_pursuit
from src.lib.approximation.sparse import multi_warm_js_subspace_pursuit
from src.lib.approximation.sparse import orthogonal_matching_pursuit
from src.lib.approximation.sparse import subspace_pursuit
from src.lib.approximation.sparse import warm_compressive_sampling_matching_pursuit
from src.lib.approximation.sparse import warm_js
from src.lib.approximation.sparse import warm_kl
from src.lib.approximation.sparse import warm_kl_compressive_sampling_matching_pursuit


def _cases():
    return [
        (
            divergence.total_variation,
            lambda *problem: brute_force_search.solve(
                *problem,
                solve_dense=dense.total_variation,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: compressive_sampling_matching_pursuit.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                normalize=normalize.clip,
                L=[2 * k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: forward_backward_pursuit.solve(
                *problem,
                solve_dense=dense.total_variation,
                normalize=normalize.clip,
                alpha=2,
                beta=1,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: frank_wolfe.solve(
                *problem,
                solve_dense=dense.total_variation,
                normalize=normalize.clip,
                is_step_size_adaptive=False,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: frank_wolfe.solve(
                *problem,
                solve_dense=dense.total_variation,
                normalize=normalize.clip,
                is_step_size_adaptive=True,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: generalized_orthogonal_matching_pursuit.solve(
                *problem,
                solve_dense=dense.total_variation,
                normalize=normalize.clip,
                L=1,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: generalized_reverse_matching_pursuit.solve(
                *problem,
                solve_dense=dense.total_variation,
                L=1,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: multi_warm_js_subspace_pursuit.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                etas=[1 / (2 * k)],
                I={2 * k},
                L=[k],
            ),
        ),
        (
            divergence.euclidean,
            lambda *problem: orthogonal_matching_pursuit.solve(
                *problem,
                solve_dense=dense.euclidean,
                normalize=lambda r: r,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: orthogonal_matching_pursuit.solve(
                *problem,
                solve_dense=dense.total_variation,
                normalize=normalize.clip,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: subspace_pursuit.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                normalize=normalize.clip,
                L=[k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: warm_compressive_sampling_matching_pursuit.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=-2,
                I=k,
                normalize=normalize.clip,
                L=[2 * k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: warm_compressive_sampling_matching_pursuit.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=None,
                I=k,
                normalize=normalize.clip,
                L=[2 * k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: warm_js.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=1 / (2 * k),
                I=2 * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: warm_js.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=None,
                I=2 * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: warm_kl_compressive_sampling_matching_pursuit.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=-2,
                I=k,
                L=[2 * k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: warm_kl_compressive_sampling_matching_pursuit.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=None,
                I=k,
                L=[2 * k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: warm_kl.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=-2,
                I=k,
            ),
        ),
        (
            divergence.total_variation,
            lambda A, b, D, k: warm_kl.solve(
                A,
                b,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=None,
                I=k,
            ),
        ),
    ]


@mark.parametrize("D,solve", _cases())
def test_selects_single_atom(D, solve):
    A = numpy.array([[1, 0.5, 0.4, 0.8], [0, 0.5, 0.6, 0.2]])
    b = numpy.array([0, 1])
    k = 1

    x = solve(A, b, D, k)

    assert numpy.nonzero(x) == numpy.array([2])
