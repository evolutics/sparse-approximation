import numpy

from src.lib import sorting
from src.lib.approximation.sparse.common import identification


def solve(C, p, D, k, *, solve_dense, L):
    m, n = C.shape
    S = numpy.full(n, False)
    q = numpy.zeros(m)

    for l in L:
        spaces = identification.shift(C=C, p=p, D=D, q=q)
        S[sorting.argmins(spaces, l)] = True

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(y, k)] = True

        q = C[:, S] @ y[S]

    y = numpy.zeros(n)
    y[S] = solve_dense(C[:, S], p)

    return y
