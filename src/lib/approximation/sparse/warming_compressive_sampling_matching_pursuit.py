import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm_kl
from src.lib.approximation.sparse.common import warm


def solve(C, p, D, k, *, solve_dense, eta, is_kl_not_js, j, L):
    n = C.shape[1]

    best_y = warm_kl.solve(C, p, D, k, solve_dense=solve_dense, eta=eta, j=j)
    S = best_y != 0
    q = C[:, S] @ best_y[S]
    best_divergence = D(p, q)

    for l in L:
        ys = warm.iterate(C=C, p=p, D=D, eta=eta, is_kl_not_js=is_kl_not_js, q=q)
        y = next(y for i, y in enumerate(ys) if numpy.count_nonzero(y) >= l or i >= j)
        S |= y != 0

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(y, k)] = True

        q = C[:, S] @ y[S]

    y = numpy.zeros(n)
    y[S] = solve_dense(C[:, S], p)
    divergence = D(p, C[:, S] @ y[S])
    if divergence < best_divergence:
        best_y = y

    return best_y
