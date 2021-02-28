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

        index = numpy.argmin(potentials)
        S[numpy.flatnonzero(~S)[index]] = True

        q = Q[:, index]

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x
