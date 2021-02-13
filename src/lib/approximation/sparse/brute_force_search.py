import itertools
import math

import numpy


def solve(A, b, D, K, solve_dense):
    N = A.shape[1]

    x_divergence = math.inf

    for combination in itertools.combinations(range(N), K):
        S = numpy.array(combination)

        z = solve_dense(A[:, S], b)
        z_divergence = D(b, A[:, S] @ z)

        if z_divergence < x_divergence:
            x = numpy.zeros(N)
            x[S] = z
            x_divergence = z_divergence

    return x
