import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm_kl


def solve(C, p, D, k, *, solve_dense, eta, I, normalize, L):
    n = C.shape[1]

    best_y = warm_kl.solve(C, p, D, k, solve_dense=solve_dense, eta=eta, I=I)
    S = best_y != 0
    q = C[:, S] @ best_y[S]
    best_divergence = D(p, q)
    r = p - q

    for l in L:
        potentials = D(normalize(r), C)
        S[sorting.argmins(potentials, l)] = True

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(y, k)] = True

        r = p - C[:, S] @ y[S]

    y = numpy.zeros(n)
    y[S] = solve_dense(C[:, S], p)
    divergence = D(p, C[:, S] @ y[S])
    if divergence < best_divergence:
        best_y = y

    return best_y
