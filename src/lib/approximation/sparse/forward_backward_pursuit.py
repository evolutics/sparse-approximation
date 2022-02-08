import numpy

from src.lib import sorting
from src.lib.approximation.sparse.common import identification


def solve(C, p, D, k, *, solve_dense, alpha, beta):
    m, n = C.shape
    S = numpy.full(n, False)
    q = numpy.zeros(m)

    while numpy.count_nonzero(S) < k:
        spaces = identification.shift(C=C[:, ~S], p=p, D=D, q=q)
        T = numpy.flatnonzero(~S)[sorting.argmins(spaces, alpha)]
        S[T] = True

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        count = max(beta, numpy.count_nonzero(S) - k)
        T = numpy.flatnonzero(S)[sorting.argmins(y[S], count)]
        S[T] = False

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        q = C[:, S] @ y[S]

    return y
