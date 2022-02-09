from pytest import mark
import numpy

from src.lib import divergence
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
from src.lib.approximation.sparse import warm_kl_like
from src.lib.approximation.sparse import warming_compressive_sampling_matching_pursuit


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
            lambda C, p, D, k: compressive_sampling_matching_pursuit.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                L=[2 * k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: forward_backward_pursuit.solve(
                *problem,
                solve_dense=dense.total_variation,
                alpha=2,
                beta=1,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: frank_wolfe.solve(
                *problem,
                solve_dense=dense.total_variation,
                is_step_size_adaptive=False,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: frank_wolfe.solve(
                *problem,
                solve_dense=dense.total_variation,
                is_step_size_adaptive=True,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: generalized_orthogonal_matching_pursuit.solve(
                *problem,
                solve_dense=dense.total_variation,
                l=1,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: generalized_reverse_matching_pursuit.solve(
                *problem,
                solve_dense=dense.total_variation,
                l=1,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: multi_warm_js_subspace_pursuit.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                etas=[1 / (2 * k)],
                J={2 * k},
                L=[k],
            ),
        ),
        (
            divergence.euclidean,
            lambda *problem: orthogonal_matching_pursuit.solve(
                *problem,
                solve_dense=dense.euclidean,
            ),
        ),
        (
            divergence.total_variation,
            lambda *problem: orthogonal_matching_pursuit.solve(
                *problem,
                solve_dense=dense.total_variation,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: subspace_pursuit.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                L=[k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: warm_compressive_sampling_matching_pursuit.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=-2,
                j=k,
                L=[2 * k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: warm_compressive_sampling_matching_pursuit.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=None,
                j=k,
                L=[2 * k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: warm_kl_like.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=-2,
                is_kl_not_js=False,
                j=k,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: warm_kl_like.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=None,
                is_kl_not_js=False,
                j=k,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: warm_kl_like.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=-2,
                is_kl_not_js=True,
                j=k,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: warm_kl_like.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=None,
                is_kl_not_js=True,
                j=k,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: warming_compressive_sampling_matching_pursuit.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=-2,
                is_kl_not_js=True,
                j=k,
                L=[2 * k] * k,
            ),
        ),
        (
            divergence.total_variation,
            lambda C, p, D, k: warming_compressive_sampling_matching_pursuit.solve(
                C,
                p,
                D,
                k,
                solve_dense=dense.total_variation,
                eta=None,
                is_kl_not_js=True,
                j=k,
                L=[2 * k] * k,
            ),
        ),
    ]


@mark.parametrize("D,solve", _cases())
def test_selects_single_atom(D, solve):
    C = numpy.array([[1, 0.5, 0.4, 0.8], [0, 0.5, 0.6, 0.2]])
    p = numpy.array([0, 1])
    k = 1

    y = solve(C, p, D, k)

    assert numpy.nonzero(y) == numpy.array([2])
