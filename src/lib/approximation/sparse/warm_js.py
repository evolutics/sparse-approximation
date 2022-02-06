import itertools

from numpy import ma
import numpy

from src.lib import sorting


def solve(A, b, D, K, *, solve_dense, eta, I):
    N = A.shape[1]

    xs_ = iterate(A=A, b=b, D=D, eta=eta)
    x = next(itertools.islice(xs_, I, None))

    S = numpy.full(N, False)
    S[sorting.argmaxs(x, K)] = True

    x = numpy.zeros(N)
    x[S] = solve_dense(A[:, S], b)

    return x


def iterate(*, A, b, D, eta):
    M, N = A.shape

    x = numpy.zeros(N)
    q = numpy.zeros(M)

    yield x

    for i in itertools.count():
        if eta is None:
            eta_i = D(b, q)
        elif eta >= 0:
            eta_i = eta
        else:
            eta_i = 1 / (-eta * i + 1)
        Q = (1 - eta_i) * q[:, None] + eta_i * A

        index = numpy.argmin(_optimized_js_divergences(b, Q))

        x *= 1 - eta_i
        x[index] += eta_i
        q = Q[:, index]

        yield x


def _optimized_js_divergences(b, Q):
    pi_ = 0.5

    q_log_q = _x_log_x(Q)
    pi_sum_q_log_q = pi_ * numpy.sum(q_log_q, axis=0)

    combined = (1 - pi_) * b[:, None] + pi_ * Q
    combined_log_combined = _x_log_x(combined)
    sum_combined_log_combined = numpy.sum(combined_log_combined, axis=0)

    return pi_sum_q_log_q - sum_combined_log_combined


def _x_log_x(x):
    return x * ma.log(x).filled(0)
