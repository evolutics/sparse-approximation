import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm_kl_like
from src.lib.approximation.sparse.common import identification


def solve(C, p, D, k, *, solve_dense, eta, is_kl_not_js, j, L):
    n = C.shape[1]

    best_y = warm_kl_like.solve(
        C, p, D, k, solve_dense=solve_dense, eta=eta, is_kl_not_js=is_kl_not_js, j=j
    )
    S = best_y != 0
    q = C[:, S] @ best_y[S]
    best_divergence = D(p, q)

    for l in L:
        spaces = identification.shift(C=C, p=p, D=D, q=q)
        S[sorting.argmins(spaces, l)] = True

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
