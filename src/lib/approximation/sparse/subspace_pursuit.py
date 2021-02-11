import numpy

from src.lib import sorting


def solve(A, b, K, D, solve_dense, normalize, I, L):
    N = A.shape[1]
    T = numpy.full(N, False)
    r = b

    for i in range(I):
        potentials = D(normalize(r), A)
        T[sorting.argmins(potentials, L)] = True

        x = numpy.zeros(N)
        x[T] = solve_dense(A[:, T], b)

        T.fill(False)
        T[sorting.argmaxs(x, K)] = True

        x = numpy.zeros(N)
        x[T] = solve_dense(A[:, T], b)

        y = A[:, T] @ x[T]
        divergence = D(b, y)

        if i == 0 or divergence < best_divergence:
            S = numpy.copy(T)
            best_divergence = divergence

        r = b - y

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x
