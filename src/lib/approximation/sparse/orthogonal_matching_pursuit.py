import numpy


def solve(A, b, K, solve_dense, potential):
    N = A.shape[1]
    S = numpy.full(N, False)
    r = b

    while numpy.count_nonzero(S) < K:
        potentials = potential(r, A[:, ~S])
        n = numpy.argwhere(~S)[numpy.argmin(potentials)][0]
        S[n] = True

        x = numpy.zeros(N)
        x[S] = solve_dense(A[:, S], b)

        r = b - A[:, S] @ x[S]

    return x
