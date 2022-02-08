import numpy

from src.lib.approximation.sparse.common import identification


def solve(C, p, D, k, *, solve_dense, is_step_size_adaptive):
    m, n = C.shape
    S = numpy.full(n, False)
    q = numpy.zeros(m)

    for i in range(k):
        spaces = identification.shift(C=C[:, ~S], p=p, D=D, q=q)
        index = numpy.flatnonzero(~S)[numpy.argmin(spaces)]
        S[index] = True

        if i == 0:
            q = C[:, index]
        else:
            if is_step_size_adaptive:
                step_size = solve_dense(numpy.column_stack([q, C[:, index]]), p)[1]
            else:
                step_size = 2 / (i + 2)
            q = (1 - step_size) * q + step_size * C[:, index]

    y = numpy.zeros(n)
    y[S] = solve_dense(C[:, S], p)

    return y
