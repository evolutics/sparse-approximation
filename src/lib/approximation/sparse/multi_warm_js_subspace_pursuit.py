import itertools
import math

import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm


def solve(C, p, D, k, *, solve_dense, etas, iterations, L):
    n = C.shape[1]

    best_divergence = math.inf
    solve_dense_cache = {}

    def cached_solve_dense(S):
        indices = frozenset(numpy.nonzero(S)[0])
        try:
            return solve_dense_cache[indices]
        except KeyError:
            y = solve_dense(C[:, S], p)
            solve_dense_cache[indices] = y
            return y

    for eta in etas:
        ys = warm.iterate(C=C, p=p, D=D, eta=eta, is_kl_not_js=False, q=None)

        for y in itertools.islice(
            (y for i, y in enumerate(ys) if i in iterations), len(iterations)
        ):
            S = numpy.full(n, False)
            S[sorting.argmaxs(y, k)] = True
            y = numpy.zeros(n)
            y[S] = cached_solve_dense(S)

            q = C[:, S] @ y[S]
            divergence = D(p, q)
            if divergence < best_divergence:
                best_y = y
                best_divergence = divergence

            for l in L:
                r = p - q
                r_normalized = r / numpy.sum(numpy.abs(r))
                shift = C - q[:, None]
                shift_normalized = shift / numpy.sum(numpy.abs(shift), axis=0)
                divergences = D(r_normalized, shift_normalized)

                S[sorting.argmins(divergences, l)] = True
                y = numpy.zeros(n)
                y[S] = cached_solve_dense(S)

                S.fill(False)
                S[sorting.argmaxs(y, k)] = True
                y = numpy.zeros(n)
                y[S] = cached_solve_dense(S)

                q = C[:, S] @ y[S]
                divergence = D(p, q)
                if divergence < best_divergence:
                    best_y = y
                    best_divergence = divergence

    return best_y
