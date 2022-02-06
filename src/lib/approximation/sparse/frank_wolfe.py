import numpy


def solve(A, b, D, K, *, solve_dense, normalize, is_step_size_adaptive):
    N = A.shape[1]
    S = numpy.full(N, False)
    r = b

    for i in range(K):
        potentials = D(normalize(r), A[:, ~S])
        index = numpy.flatnonzero(~S)[numpy.argmin(potentials)]
        S[index] = True

        if i == 0:
            y = A[:, index]
        else:
            if is_step_size_adaptive:
                step_size = solve_dense(numpy.column_stack([y, A[:, index]]), b)[1]
            else:
                step_size = 2 / (i + 2)
            y = (1 - step_size) * y + step_size * A[:, index]

        r = b - y

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x
