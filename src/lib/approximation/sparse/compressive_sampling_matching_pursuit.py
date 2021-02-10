import numpy

from src.lib import sorting


def solve(A, b, K, D, solve_dense, normalize, I, L):
    N = A.shape[1]
    S = numpy.full(N, False)
    r = b

    for _ in range(I):
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, L)] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, K)] = True

        x[~S] = 0
        x = normalize(x)

        r = b - A[:, S] @ x[S]

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x
