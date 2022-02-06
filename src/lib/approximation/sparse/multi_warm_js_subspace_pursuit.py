import itertools
import math

import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm


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
        xs_ = warm.iterate(A=A, b=b, D=D, eta=eta, is_kl_not_js=False, q=None)

        for x in itertools.islice((x for i, x in enumerate(xs_) if i in I), len(I)):
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
