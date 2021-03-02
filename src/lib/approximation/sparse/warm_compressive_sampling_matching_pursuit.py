import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm_kl


def solve(A, b, D, K, solve_dense, eta_i, normalize, I, L):
    N = A.shape[1]

    z = warm_kl.solve(A, b, D, K, solve_dense, eta_i)
    S = z != 0
    y = A[:, S] @ z[S]
    best_divergence = D(b, y)
    r = b - y

    for _ in range(I):
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, L)] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, K)] = True

        y = A[:, S] @ x[S]
        divergence = D(b, y)

        if divergence < best_divergence:
            x[~S] = 0
            z = x
            best_divergence = divergence

        r = b - y

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)
    divergence = D(b, A[:, S] @ x[S])
    if divergence < best_divergence:
        z = x

    return z
