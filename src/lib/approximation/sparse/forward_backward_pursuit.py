import numpy

from src.lib import sorting


def solve(A, b, D, k, *, solve_dense, normalize, alpha, beta):
    N = A.shape[1]
    S = numpy.full(N, False)
    r = b

    while numpy.count_nonzero(S) < k:
        potentials = D(normalize(r), A[:, ~S])
        T = numpy.flatnonzero(~S)[sorting.argmins(potentials, alpha)]
        S[T] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        count = max(beta, numpy.count_nonzero(S) - k)
        T = numpy.flatnonzero(S)[sorting.argmins(x[S], count)]
        S[T] = False

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        r = b - A[:, S] @ x[S]

    return x
