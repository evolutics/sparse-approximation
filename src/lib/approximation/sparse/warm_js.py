import itertools

from numpy import ma
import numpy


def iterate(*, A, b, eta_i):
    M, N = A.shape

    q = numpy.zeros(M)
    x = numpy.zeros(N)

    for i in itertools.count():
        eta = eta_i(i)
        Q = (1 - eta) * q[:, None] + eta * A

        index = numpy.argmin(_optimized_js_divergences(b, Q))

        q = Q[:, index]
        x *= 1 - eta
        x[index] += eta

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
