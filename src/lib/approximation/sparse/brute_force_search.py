import itertools
import math

import numpy


def solve(C, p, D, k, *, solve_dense):
    n = C.shape[1]

    best_divergence = math.inf

    for combination in itertools.combinations(range(n), k):
        S = numpy.array(combination)

        y = solve_dense(C[:, S], p)
        divergence = D(p, C[:, S] @ y)

        if divergence < best_divergence:
            best_y = numpy.zeros(n)
            best_y[S] = y
            best_divergence = divergence

    return best_y
