import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm_kl


def solve(A, b, D, K, solve_dense, eta_i, normalize, I, L):
    N = A.shape[1]

    x = warm_kl.solve(A, b, D, K, solve_dense, eta_i)
    S = x != 0
    y = A[:, S] @ x[S]
    T = numpy.copy(S)
    best_divergence = D(b, y)
    r = b - y

    for _ in range(I):
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, L)] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, K)] = True

        x[~S] = 0

        y = A[:, S] @ x[S]
        divergence = D(b, y)

        if divergence < best_divergence:
            T = numpy.copy(S)
            best_divergence = divergence

        r = b - y

    x = numpy.zeros(N)
    x[T] = solve_dense(A[:, T], b)

    return x
