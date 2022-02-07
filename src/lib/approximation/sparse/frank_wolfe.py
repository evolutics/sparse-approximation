import numpy


def solve(A, b, D, k, *, solve_dense, normalize, is_step_size_adaptive):
    n = A.shape[1]
    S = numpy.full(n, False)
    r = b

    for i in range(k):
        potentials = D(normalize(r), A[:, ~S])
        index = numpy.flatnonzero(~S)[numpy.argmin(potentials)]
        S[index] = True

        if i == 0:
            q = A[:, index]
        else:
            if is_step_size_adaptive:
                step_size = solve_dense(numpy.column_stack([q, A[:, index]]), b)[1]
            else:
                step_size = 2 / (i + 2)
            q = (1 - step_size) * q + step_size * A[:, index]

        r = b - q

    x = numpy.zeros(n)
    x[S] = solve_dense(A[:, S], b)

    return x
