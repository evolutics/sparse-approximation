import math

import numpy

from src.lib import sorting


def solve(A, b, D, k, *, solve_dense, normalize, L):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = b
    best_divergence = math.inf

    for l in L:
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, l)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, k)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], b)

        y = A[:, S] @ x[S]
        divergence = D(b, y)

        if divergence < best_divergence:
            best_x = x
            best_divergence = divergence

        r = b - y

    return best_x
