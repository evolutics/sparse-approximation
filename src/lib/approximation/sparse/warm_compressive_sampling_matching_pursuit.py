import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm_kl


def solve(A, p, D, k, *, solve_dense, eta, I, normalize, L):
    n = A.shape[1]

    best_x = warm_kl.solve(A, p, D, k, solve_dense=solve_dense, eta=eta, I=I)
    S = best_x != 0
    q = A[:, S] @ best_x[S]
    best_divergence = D(p, q)
    r = p - q

    for l in L:
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, l)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(x, k)] = True

        r = p - A[:, S] @ x[S]

    x = numpy.zeros(n)
    x[S] = solve_dense(A[:, S], p)
    divergence = D(p, A[:, S] @ x[S])
    if divergence < best_divergence:
        best_x = x

    return best_x
