import numpy

from src.lib import sorting


def solve(C, p, D, k, *, solve_dense, normalize, L):
    n = C.shape[1]
    S = numpy.full(n, False)
    r = p

    while numpy.count_nonzero(S) < k:
        potentials = D(normalize(r), C[:, ~S])
        count = min(L, k - numpy.count_nonzero(S))
        T = numpy.flatnonzero(~S)[sorting.argmins(potentials, count)]
        S[T] = True

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        r = p - C[:, S] @ y[S]

    return y
