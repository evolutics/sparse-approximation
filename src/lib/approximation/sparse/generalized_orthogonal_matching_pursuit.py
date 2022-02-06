import numpy

from src.lib import sorting


def solve(A, b, D, k, *, solve_dense, normalize, L):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = b

    while numpy.count_nonzero(S) < k:
        potentials = D(normalize(r), A[:, ~S])
        count = min(L, k - numpy.count_nonzero(S))
        T = numpy.flatnonzero(~S)[sorting.argmins(potentials, count)]
        S[T] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], b)

        r = b - A[:, S] @ x[S]

    return x
