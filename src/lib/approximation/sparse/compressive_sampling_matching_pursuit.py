import numpy

from src.lib import sorting


def solve(A, b, D, k, *, solve_dense, normalize, L):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = b

    for l in L:
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, l)] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, k)] = True

        r = b - A[:, S] @ x[S]

    x = numpy.zeros(n)
    x[S] = solve_dense(A[:, S], b)

    return x
