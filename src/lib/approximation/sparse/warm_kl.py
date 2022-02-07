import numpy

from src.lib.approximation.sparse import warm


def solve(C, p, D, k, *, solve_dense, eta, I):
    n = C.shape[1]

    xs_ = warm.iterate(C=C, p=p, D=D, eta=eta, is_kl_not_js=True, q=None)
    x = next(x for i, x in enumerate(xs_) if numpy.count_nonzero(x) >= k or i >= I)
    S = x != 0

    x = numpy.zeros(n)
    x[S] = solve_dense(C[:, S], p)

    return x
