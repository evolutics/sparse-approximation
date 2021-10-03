import itertools
import math

import numpy


def solve(A, b, D, K, *, solve_dense):
    N = A.shape[1]

    best_divergence = math.inf

    for combination in itertools.combinations(range(N), K):
        S = numpy.array(combination)

        x = solve_dense(A[:, S], b)
        divergence = D(b, A[:, S] @ x)

        if divergence < best_divergence:
            best_x = numpy.zeros(N)
            best_x[S] = x
            best_divergence = divergence

    return best_x
