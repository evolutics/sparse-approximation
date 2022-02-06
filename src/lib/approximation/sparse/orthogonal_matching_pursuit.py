import numpy


def solve(A, b, D, k, *, solve_dense, normalize):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = b

    while numpy.count_nonzero(S) < k:
        potentials = D(normalize(r), A[:, ~S])
        index = numpy.flatnonzero(~S)[numpy.argmin(potentials)]
        S[index] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], b)

        r = b - A[:, S] @ x[S]

    return x
