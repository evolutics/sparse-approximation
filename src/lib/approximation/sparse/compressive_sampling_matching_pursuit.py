import numpy

from src.lib import sorting


def solve(A, b, D, K, *, solve_dense, normalize, L):
    N = A.shape[1]
    S = numpy.full(N, False)
    r = b

    for l in L:
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, l)] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, K)] = True

        r = b - A[:, S] @ x[S]

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x
