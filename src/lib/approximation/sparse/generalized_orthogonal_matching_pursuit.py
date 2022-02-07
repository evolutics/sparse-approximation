import numpy

from src.lib import sorting


def solve(A, p, D, k, *, solve_dense, normalize, L):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = p

    while numpy.count_nonzero(S) < k:
        potentials = D(normalize(r), A[:, ~S])
        count = min(L, k - numpy.count_nonzero(S))
        T = numpy.flatnonzero(~S)[sorting.argmins(potentials, count)]
        S[T] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

        r = p - A[:, S] @ x[S]

    return x
