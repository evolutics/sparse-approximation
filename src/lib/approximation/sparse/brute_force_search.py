import itertools
import math

import numpy


def solve(C, p, D, k, *, solve_dense):
    n = C.shape[1]

    best_divergence = math.inf

    for combination in itertools.combinations(range(n), k):
        S = numpy.array(combination)

        x = solve_dense(C[:, S], p)
        divergence = D(p, C[:, S] @ x)

        if divergence < best_divergence:
            best_x = numpy.zeros(n)
            best_x[S] = x
            best_divergence = divergence

    return best_x
