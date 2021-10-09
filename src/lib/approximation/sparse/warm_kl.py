# pylint: disable=unsubscriptable-object

import numpy


def solve(A, b, D, K, *, solve_dense, eta_i, I):
    S = select(A, b, D, K, eta_i=eta_i, I=I, q=None)

    N = A.shape[1]
    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x


def select(A, b, D, K, *, eta_i, I, q):
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
            if eta_i is None:
                eta = D(b, q)
            else:
                eta = eta_i(i)
            Q = (1 - eta) * q[:, None] + eta * A

        # A `q` minimizes the K directed divergence `K(b, q)` if and only if it
        # maximizes `∑ₘ bₘ log (bₘ+qₘ)`, which is faster to compute.
        potentials = b @ numpy.log(b[:, None] + Q)
        index = numpy.argmax(potentials)

        S[index] = True
        if numpy.count_nonzero(S) == K:
            break

        q = Q[:, index]

    return S
