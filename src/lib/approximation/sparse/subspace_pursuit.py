import math

import numpy

from src.lib import sorting


def solve(A, b, D, K, solve_dense, normalize, L):
    N = A.shape[1]
    S = numpy.full(N, False)
    r = b
    best_divergence = math.inf

    for l in L:
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, l)] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, K)] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        y = A[:, S] @ x[S]
        divergence = D(b, y)

        if divergence < best_divergence:
            best_x = x
            best_divergence = divergence

        r = b - y

    return best_x
