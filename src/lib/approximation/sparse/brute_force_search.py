import itertools
import math

import numpy


def solve(A, b, K, D, solve_dense):
    N = A.shape[1]

    x_divergence = math.inf

    for S in itertools.combinations(range(N), K):
        z = solve_dense(A[:, S], b)
        z_divergence = D(b, A[:, S] @ z)

        if z_divergence < x_divergence:
            x = numpy.zeros(N)
            x[S] = z
            x_divergence = z_divergence

    return x
