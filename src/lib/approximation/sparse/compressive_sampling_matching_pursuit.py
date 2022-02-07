import numpy

from src.lib import sorting


def solve(A, p, D, k, *, solve_dense, normalize, L):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = p

    for l in L:
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, l)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(x, k)] = True

        r = p - A[:, S] @ x[S]

    x = numpy.zeros(n)
    x[S] = solve_dense(A[:, S], p)

    return x
