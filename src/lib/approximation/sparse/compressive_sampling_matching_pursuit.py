import numpy

from src.lib import sorting


def solve(C, p, D, k, *, solve_dense, normalize, L):
    n = C.shape[1]
    S = numpy.full(n, False)
    r = p

    for l in L:
        potentials = D(normalize(r), C)
        S[sorting.argmins(potentials, l)] = True

        y = numpy.zeros(n)
        y[S] = solve_dense(C[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(y, k)] = True

        r = p - C[:, S] @ y[S]

    y = numpy.zeros(n)
    y[S] = solve_dense(C[:, S], p)

    return y
