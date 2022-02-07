import numpy


def solve(A, p, D, k, *, solve_dense, normalize):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = p

    while numpy.count_nonzero(S) < k:
        potentials = D(normalize(r), A[:, ~S])
        index = numpy.flatnonzero(~S)[numpy.argmin(potentials)]
        S[index] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

        r = p - A[:, S] @ x[S]

    return x
