import numpy

from src.lib import sorting


def solve(A, p, _, k, *, solve_dense, L):
    n = A.shape[1]
    S = numpy.full(n, True)
    x = solve_dense(A, p)

    while numpy.count_nonzero(S) > k:
        surplus = numpy.count_nonzero(S) - k
        l = L if isinstance(L, int) else L(surplus)
        drops = max(min(l, surplus), 1)
        T = numpy.flatnonzero(S)[sorting.argmins(x[S], drops)]
        S[T] = False

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

    return x
