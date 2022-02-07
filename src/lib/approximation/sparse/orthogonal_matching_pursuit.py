import numpy


def solve(C, p, D, k, *, solve_dense, normalize):
    n = C.shape[1]
    S = numpy.full(n, False)
    r = p

    while numpy.count_nonzero(S) < k:
        potentials = D(normalize(r), C[:, ~S])
        index = numpy.flatnonzero(~S)[numpy.argmin(potentials)]
        S[index] = True

        x = numpy.zeros(n)
        x[S] = solve_dense(C[:, S], p)

        r = p - C[:, S] @ x[S]

    return x
