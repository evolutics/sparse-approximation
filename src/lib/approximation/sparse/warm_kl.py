import numpy


def solve(A, b, _, K, solve_dense, eta_i):
    N = A.shape[1]
    S = numpy.full(N, False)

    nonzero = b != 0
    b = b[nonzero]
    A = A[nonzero, :]

    for i in range(K):
        if i == 0:
            Q = A
        else:
            eta = eta_i(i)
            Q = (1 - eta) * q[:, None] + eta * A[:, ~S]

        # A `q` minimizes the K directed divergence `K(b, q)` if and only if it
        # maximizes `∑ₘ bₘ log (bₘ+qₘ)`, which is faster to compute.
        potentials = b @ numpy.log(b[:, None] + Q)
        index = numpy.argmax(potentials)

        S[numpy.flatnonzero(~S)[index]] = True

        q = Q[:, index]

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x
