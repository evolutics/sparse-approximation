import numpy

from src.lib import divergence


def solve(A, b, _, K, solve_dense, eta_i):
    N = A.shape[1]
    S = numpy.full(N, False)

    for i in range(K):
        if i == 0:
            Q = A
        else:
            eta = eta_i(i)
            Q = (1 - eta) * q[:, None] + eta * A[:, ~S]
        potentials = divergence.k_directed(b, Q)

        n = numpy.flatnonzero(~S)[numpy.argmin(potentials)]
        S[n] = True

        if i == 0:
            q = A[:, n]
        else:
            q = (1 - eta) * q + eta * A[:, n]

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x
