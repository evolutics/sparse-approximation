import itertools

from numpy import ma
import numpy


def iterate(*, C, p, D, eta, is_kl_not_js, q):
    m, n = C.shape

    y = numpy.zeros(n)
    if q is None:
        q = numpy.zeros(m)

    yield y

    for i in itertools.count():
        if eta is None:
            eta_i = D(p, q)
        elif eta >= 0:
            eta_i = eta / (1 - (1 - eta) ** (i + 1))
        else:
            eta_i = 1 / (-eta * i + 1)
        Q = (1 - eta_i) * q[:, None] + eta_i * C

        index = numpy.argmin(
            _optimized_k_directed_divergences(p, Q)
            if is_kl_not_js
            else _optimized_js_divergences(p, Q)
        )

        y *= 1 - eta_i
        y[index] += eta_i
        q = Q[:, index]

        yield y


def _optimized_k_directed_divergences(p, Q):
    # A `q` minimizes the K directed divergence `K(p, q)` if and only if it
    # minimizes `-∑ᵢ pᵢ log (pᵢ+qᵢ)`, which is faster to compute.

    nonzero = p != 0
    p = p[nonzero]
    Q = Q[nonzero, :]

    return -p @ numpy.log(p[:, None] + Q)


def _optimized_js_divergences(p, Q):
    pi_ = 0.5

    q_log_q = _x_log_x(Q)
    pi_sum_q_log_q = pi_ * numpy.sum(q_log_q, axis=0)

    combined = (1 - pi_) * p[:, None] + pi_ * Q
    combined_log_combined = _x_log_x(combined)
    sum_combined_log_combined = numpy.sum(combined_log_combined, axis=0)

    return pi_sum_q_log_q - sum_combined_log_combined


def _x_log_x(values):
    return values * ma.log(values).filled(0)
