import math

import numpy

from src.lib import sorting
from src.lib.approximation.sparse.common import identification


def solve(C, p, D, k, *, solve_dense, L):
    m, n = C.shape
    S = numpy.full(n, False)
    q = numpy.zeros(m)
    best_divergence = math.inf

    for l in L:
        spaces = identification.shift(C=C, p=p, D=D, q=q)
        S[sorting.argmins(spaces, l)] = True

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(y, k)] = True

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        q = C[:, S] @ y[S]
        divergence = D(p, q)

        if divergence < best_divergence:
            best_y = y
            best_divergence = divergence

    return best_y
