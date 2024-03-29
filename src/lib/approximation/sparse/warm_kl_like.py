import numpy

from src.lib.approximation.sparse.common import warm


def solve(C, p, D, k, *, solve_dense, eta, is_kl_not_js, j):
    n = C.shape[1]

    ys = warm.iterate(C=C, p=p, D=D, eta=eta, is_kl_not_js=is_kl_not_js, q=None)
    y = next(y for i, y in enumerate(ys) if numpy.count_nonzero(y) >= k or i >= j)
    S = y != 0

    y = numpy.zeros(n)
    y[S] = solve_dense(C[:, S], p)

    return y
