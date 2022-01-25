import numpy

from src.lib import sorting


def solve(A, b, _, K, *, solve_dense, L):
    N = A.shape[1]
    S = numpy.full(N, True)
    x = solve_dense(A, b)

    while numpy.count_nonzero(S) > K:
        surplus = numpy.count_nonzero(S) - K
        drops = min(L, surplus)
        T = numpy.flatnonzero(S)[sorting.argmins(x[S], drops)]
        S[T] = False

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

    return x
