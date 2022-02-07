import math

import numpy

from src.lib import sorting


def solve(C, p, D, k, *, solve_dense, normalize, L):
    n = C.shape[1]
    S = numpy.full(n, False)
    r = p
    best_divergence = math.inf

    for l in L:
        potentials = D(normalize(r), C)
        S[sorting.argmins(potentials, l)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(C[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(x, k)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(C[:, S], p)

        q = C[:, S] @ x[S]
        divergence = D(p, q)

        if divergence < best_divergence:
            best_x = x
            best_divergence = divergence

        r = p - q

    return best_x
