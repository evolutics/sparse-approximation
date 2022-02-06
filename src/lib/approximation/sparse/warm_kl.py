# pylint: disable=unsubscriptable-object

import itertools

import numpy


def solve(A, b, D, K, *, solve_dense, eta, I):
    N = A.shape[1]

    xs_ = iterate(A=A, b=b, D=D, eta=eta, q=None)
    x = next(x for i, x in enumerate(xs_) if numpy.count_nonzero(x) >= K or i >= I)
    S = x != 0

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x


def iterate(*, A, b, D, eta, q):
    M, N = A.shape

    x = numpy.zeros(N)
    if q is None:
        q = numpy.zeros(M)

    yield x

    for i in itertools.count():
        if eta is None:
            eta_i = D(b, q)
        elif eta >= 0:
            eta_i = eta
        else:
            eta_i = 1 / (-eta * i + 1)
        Q = (1 - eta_i) * q[:, None] + eta_i * A

        index = numpy.argmin(_optimized_k_directed_divergences(b, Q))

        x *= 1 - eta_i
        x[index] += eta_i
        q = Q[:, index]

        yield x


def _optimized_k_directed_divergences(b, Q):
    # A `q` minimizes the K directed divergence `K(b, q)` if and only if it
    # minimizes `-∑ₘ bₘ log (bₘ+qₘ)`, which is faster to compute.

    nonzero = b != 0
    b = b[nonzero]
    Q = Q[nonzero, :]

    return -b @ numpy.log(b[:, None] + Q)
