import numpy

from src.lib.approximation.sparse.common import identification


def solve(C, p, D, k, *, solve_dense):
    m, n = C.shape
    S = numpy.full(n, False)
    q = numpy.zeros(m)

    while numpy.count_nonzero(S) < k:
        spaces = identification.shift(C=C[:, ~S], p=p, D=D, q=q)
        index = numpy.flatnonzero(~S)[numpy.argmin(spaces)]
        S[index] = True

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        q = C[:, S] @ y[S]

    return y
