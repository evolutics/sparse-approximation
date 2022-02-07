import numpy

from src.lib import sorting


def solve(C, p, D, k, *, solve_dense, normalize, alpha, beta):
    n = C.shape[1]
    S = numpy.full(n, False)
    r = p

    while numpy.count_nonzero(S) < k:
        potentials = D(normalize(r), C[:, ~S])
        T = numpy.flatnonzero(~S)[sorting.argmins(potentials, alpha)]
        S[T] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(C[:, S], p)

        count = max(beta, numpy.count_nonzero(S) - k)
        T = numpy.flatnonzero(S)[sorting.argmins(x[S], count)]
        S[T] = False

        x = numpy.zeros(n)
        x[S] = solve_dense(C[:, S], p)

        r = p - C[:, S] @ x[S]

    return x
