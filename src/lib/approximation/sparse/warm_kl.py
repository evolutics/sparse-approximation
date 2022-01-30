# pylint: disable=unsubscriptable-object

import numpy


def solve(A, b, D, K, *, solve_dense, eta, I):
    S = select(A, b, D, K, eta=eta, I=I, q=None)

    N = A.shape[1]
    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x


def select(A, b, D, K, *, eta, I, q):
    M, N = A.shape

    S = numpy.full(N, False)

    if q is None:
        q = numpy.zeros(M)

    for i in range(I * K):
        if eta is None:
            eta_i = D(b, q)
        elif eta >= 0:
            eta_i = eta
        else:
            eta_i = 1 / (-eta * i + 1)
        Q = (1 - eta_i) * q[:, None] + eta_i * A

        index = numpy.argmin(_optimized_k_directed_divergences(b, Q))

        S[index] = True
        if numpy.count_nonzero(S) == K:
            break

        q = Q[:, index]

    return S


def _optimized_k_directed_divergences(b, Q):
    # A `q` minimizes the K directed divergence `K(b, q)` if and only if it
    # minimizes `-∑ₘ bₘ log (bₘ+qₘ)`, which is faster to compute.

    nonzero = b != 0
    b = b[nonzero]
    Q = Q[nonzero, :]

    return -b @ numpy.log(b[:, None] + Q)
