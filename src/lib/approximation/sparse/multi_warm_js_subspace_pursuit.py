import itertools
import math

from numpy import ma
import numpy

from src.lib import sorting


def solve(A, b, D, K, *, solve_dense, etas, I, L):
    N = A.shape[1]

    best_divergence = math.inf
    solve_dense_cache = {}

    def cached_solve_dense(S):
        indices = frozenset(numpy.nonzero(S)[0])
        try:
            return solve_dense_cache[indices]
        except KeyError:
            x = solve_dense(A[:, S], b)
            solve_dense_cache[indices] = x
            return x

    for eta in etas:
        eta_i = lambda i, eta=eta: eta if eta >= 0 else 1 / (-eta * i + 1)

        xs_ = _iterate(A=A, b=b, eta_i=eta_i)

        for x in itertools.islice((x for i, x in enumerate(xs_, 1) if i in I), len(I)):
            S = numpy.full(N, False)
            S[sorting.argmaxs(x, K)] = True
            x = numpy.zeros(N)
            x[S] = cached_solve_dense(S)

            y = A[:, S] @ x[S]
            divergence = D(b, y)
            if divergence < best_divergence:
                best_x = x
                best_divergence = divergence

            for l in L:
                r = b - y
                r_normalized = r / numpy.sum(numpy.abs(r))
                shift = A - y[:, None]
                shift_normalized = shift / numpy.sum(numpy.abs(shift), axis=0)
                divergences = D(r_normalized, shift_normalized)

                S[sorting.argmins(divergences, l)] = True
                x = numpy.zeros(N)
                x[S] = cached_solve_dense(S)

                S.fill(False)
                S[sorting.argmaxs(x, K)] = True
                x = numpy.zeros(N)
                x[S] = cached_solve_dense(S)

                y = A[:, S] @ x[S]
                divergence = D(b, y)
                if divergence < best_divergence:
                    best_x = x
                    best_divergence = divergence

    return best_x


def _iterate(*, A, b, eta_i):
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
