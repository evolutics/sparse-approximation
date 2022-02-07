import itertools
import math

import numpy


def solve(A, p, D, k, *, solve_dense):
    n = A.shape[1]

    best_divergence = math.inf

    for combination in itertools.combinations(range(n), k):
        S = numpy.array(combination)

        x = solve_dense(A[:, S], p)
        divergence = D(p, A[:, S] @ x)

        if divergence < best_divergence:
            best_x = numpy.zeros(n)
            best_x[S] = x
            best_divergence = divergence

    return best_x
