import numpy

from src.lib import sorting


def solve(A, b, _, k, *, solve_dense, L):
    N = A.shape[1]
    S = numpy.full(N, True)
    x = solve_dense(A, b)

    while numpy.count_nonzero(S) > k:
        surplus = numpy.count_nonzero(S) - k
        l = L if isinstance(L, int) else L(surplus)
        drops = max(min(l, surplus), 1)
        T = numpy.flatnonzero(S)[sorting.argmins(x[S], drops)]
        S[T] = False

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

    return x
