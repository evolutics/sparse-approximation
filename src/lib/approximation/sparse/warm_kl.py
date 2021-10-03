# pylint: disable=unsubscriptable-object

import numpy


def solve(A, b, D, K, *, solve_dense, eta_i):
    S = select(A, b, D, K, eta_i=eta_i, q=None)

    N = A.shape[1]
    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x


def select(A, b, D, K, *, eta_i, q):
    N = A.shape[1]
    S = numpy.full(N, False)

    nonzero = b != 0
    A = A[nonzero, :]
    b = b[nonzero]

    for i in range(K):
        if q is None:
            Q = A
        else:
            if eta_i is None:
                eta = D(b, q)
            else:
                eta = eta_i(i)
            Q = (1 - eta) * q[:, None] + eta * A[:, ~S]

        # A `q` minimizes the K directed divergence `K(b, q)` if and only if it
        # maximizes `∑ₘ bₘ log (bₘ+qₘ)`, which is faster to compute.
        potentials = b @ numpy.log(b[:, None] + Q)
        index = numpy.argmax(potentials)

        S[numpy.flatnonzero(~S)[index]] = True

        q = Q[:, index]

    return S
