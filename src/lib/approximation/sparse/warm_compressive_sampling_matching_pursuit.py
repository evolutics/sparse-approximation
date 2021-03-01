import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm_kl


def solve(A, b, D, K, solve_dense, eta_i, normalize, I, L):
    N = A.shape[1]

    z = warm_kl.solve(A, b, D, K, solve_dense, eta_i)
    S = z != 0
    y = A[:, S] @ z[S]
    z_divergence = D(b, y)
    r = b - y

    for _ in range(I):
        potentials = D(normalize(r), A)
        S[sorting.argmins(potentials, L)] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        S.fill(False)
        S[sorting.argmaxs(x, K)] = True

        y = A[:, S] @ x[S]
        r = b - y

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)
    x_divergence = D(b, A[:, S] @ x[S])

    if x_divergence < z_divergence:
        return x
    return z
