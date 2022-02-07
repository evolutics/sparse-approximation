import math

import numpy

from src.lib import sorting


def solve(A, p, D, k, *, solve_dense, normalize, L):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = p
    best_divergence = math.inf

    for l in L:
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, l)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(x, k)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

        q = A[:, S] @ x[S]
        divergence = D(p, q)

        if divergence < best_divergence:
            best_x = x
            best_divergence = divergence

        r = p - q

    return best_x
