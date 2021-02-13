import numpy

from src.lib import sorting


def solve(A, b, D, K, solve_dense, normalize, I, L):
    N = A.shape[1]
    S = numpy.full(N, False)
    r = b

    for i in range(I):
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, L)] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, K)] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        y = A[:, S] @ x[S]
        divergence = D(b, y)

        if i == 0 or divergence < best_divergence:
            z = x
            best_divergence = divergence

        r = b - y

    return z
