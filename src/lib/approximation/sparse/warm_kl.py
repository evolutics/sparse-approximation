# pylint: disable=unsubscriptable-object

import numpy


def solve(A, b, D, K, *, solve_dense, eta, I):
    S = select(A, b, D, K, eta=eta, I=I, q=None)

    N = A.shape[1]
    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x


def select(A, b, D, K, *, eta, I, q):
    N = A.shape[1]
    S = numpy.full(N, False)

    nonzero = b != 0
    A = A[nonzero, :]
    b = b[nonzero]
    if q is not None:
        q = q[nonzero]

    for i in range(I * K):
        if q is None:
            Q = A
        else:
            if eta is None:
                eta_i = D(b, q)
            elif eta >= 0:
                eta_i = eta
            else:
                eta_i = 1 / (-eta * i + 1)
            Q = (1 - eta_i) * q[:, None] + eta_i * A

        # A `q` minimizes the K directed divergence `K(b, q)` if and only if it
        # maximizes `∑ₘ bₘ log (bₘ+qₘ)`, which is faster to compute.
        potentials = b @ numpy.log(b[:, None] + Q)
        index = numpy.argmax(potentials)

        S[index] = True
        if numpy.count_nonzero(S) == K:
            break

        q = Q[:, index]

    return S
