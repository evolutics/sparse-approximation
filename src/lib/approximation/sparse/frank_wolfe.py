import numpy


def solve(A, b, K, solve_dense, potential, is_step_size_adaptive):
    N = A.shape[1]
    S = numpy.full(N, False)
    r = b

    for i in range(K):
        potentials = potential(r, A[:, ~S])
        n = numpy.argwhere(~S)[numpy.argmin(potentials)][0]
        S[n] = True

        if i == 0:
            y = A[:, n]
        else:
            if is_step_size_adaptive:
                step_size = solve_dense(numpy.column_stack([y, A[:, n]]), b)[1]
            else:
                step_size = 2 / (i + 2)
            y = (1 - step_size) * y + step_size * A[:, n]

        r = b - y

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x
