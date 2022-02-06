import itertools

from numpy import ma
import numpy


def iterate(*, A, b, D, eta, is_kl_not_js, q):
    M, N = A.shape

    x = numpy.zeros(N)
    if q is None:
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

        index = numpy.argmin(
            _optimized_k_directed_divergences(b, Q)
            if is_kl_not_js
            else _optimized_js_divergences(b, Q)
        )

        x *= 1 - eta_i
        x[index] += eta_i
        q = Q[:, index]

        yield x


def _optimized_k_directed_divergences(b, Q):
    # A `q` minimizes the K directed divergence `K(b, q)` if and only if it
    # minimizes `-∑ₘ bₘ log (bₘ+qₘ)`, which is faster to compute.

    nonzero = b != 0
    b = b[nonzero]
    Q = Q[nonzero, :]

    return -b @ numpy.log(b[:, None] + Q)


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
