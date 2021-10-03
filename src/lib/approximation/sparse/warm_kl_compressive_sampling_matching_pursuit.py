import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm_kl


def solve(A, b, D, K, *, solve_dense, eta_i, L):
    N = A.shape[1]

    best_x = warm_kl.solve(A, b, D, K, solve_dense=solve_dense, eta_i=eta_i)
    S = best_x != 0
    y = A[:, S] @ best_x[S]
    best_divergence = D(b, y)

    for l in L:
        S |= warm_kl.select(A, b, D, l, eta_i=eta_i, q=y)

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, K)] = True

        y = A[:, S] @ x[S]

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)
    divergence = D(b, A[:, S] @ x[S])
    if divergence < best_divergence:
        best_x = x

    return best_x
