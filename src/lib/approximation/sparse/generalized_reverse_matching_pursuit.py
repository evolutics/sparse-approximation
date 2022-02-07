import numpy

from src.lib import sorting


def solve(C, p, _, k, *, solve_dense, L):
    n = C.shape[1]
    S = numpy.full(n, True)
    y = solve_dense(C, p)

    while numpy.count_nonzero(S) > k:
        surplus = numpy.count_nonzero(S) - k
        l = L if isinstance(L, int) else L(surplus)
        drops = max(min(l, surplus), 1)
        T = numpy.flatnonzero(S)[sorting.argmins(y[S], drops)]
        S[T] = False

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

    return y
