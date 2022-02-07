import numpy

from src.lib import sorting


def solve(A, p, D, k, *, solve_dense, normalize, alpha, beta):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = p

    while numpy.count_nonzero(S) < k:
        potentials = D(normalize(r), A[:, ~S])
        T = numpy.flatnonzero(~S)[sorting.argmins(potentials, alpha)]
        S[T] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

        count = max(beta, numpy.count_nonzero(S) - k)
        T = numpy.flatnonzero(S)[sorting.argmins(x[S], count)]
        S[T] = False

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

        r = p - A[:, S] @ x[S]

    return x
