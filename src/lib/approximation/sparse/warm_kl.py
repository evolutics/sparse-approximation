import numpy

from src.lib.approximation.sparse import warm


def solve(A, b, D, k, *, solve_dense, eta, I):
    n = A.shape[1]

    xs_ = warm.iterate(A=A, b=b, D=D, eta=eta, is_kl_not_js=True, q=None)
    x = next(x for i, x in enumerate(xs_) if numpy.count_nonzero(x) >= k or i >= I)
    S = x != 0

    x = numpy.zeros(n)
    x[S] = solve_dense(A[:, S], b)

    return x
