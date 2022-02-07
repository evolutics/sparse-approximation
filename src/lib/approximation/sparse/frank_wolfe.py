import numpy


def solve(C, p, D, k, *, solve_dense, normalize, is_step_size_adaptive):
    n = C.shape[1]
    S = numpy.full(n, False)
    r = p

    for i in range(k):
        potentials = D(normalize(r), C[:, ~S])
        index = numpy.flatnonzero(~S)[numpy.argmin(potentials)]
        S[index] = True

        if i == 0:
            q = C[:, index]
        else:
            if is_step_size_adaptive:
                step_size = solve_dense(numpy.column_stack([q, C[:, index]]), p)[1]
            else:
                step_size = 2 / (i + 2)
            q = (1 - step_size) * q + step_size * C[:, index]

        r = p - q

    x = numpy.zeros(n)
    x[S] = solve_dense(C[:, S], p)

    return x
