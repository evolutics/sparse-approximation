import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm_kl


def solve(A, b, D, k, *, solve_dense, eta, I, normalize, L):
    n = A.shape[1]

    best_x = warm_kl.solve(A, b, D, k, solve_dense=solve_dense, eta=eta, I=I)
    S = best_x != 0
    y = A[:, S] @ best_x[S]
    best_divergence = D(b, y)
    r = b - y

    for l in L:
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, l)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, k)] = True

        r = b - A[:, S] @ x[S]

    x = numpy.zeros(n)
    x[S] = solve_dense(A[:, S], b)
    divergence = D(b, A[:, S] @ x[S])
    if divergence < best_divergence:
        best_x = x

    return best_x
