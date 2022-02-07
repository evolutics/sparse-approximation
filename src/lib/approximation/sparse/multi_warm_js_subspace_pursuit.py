import itertools
import math

import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm


def solve(A, p, D, k, *, solve_dense, etas, I, L):
    n = A.shape[1]

    best_divergence = math.inf
    solve_dense_cache = {}

    def cached_solve_dense(S):
        indices = frozenset(numpy.nonzero(S)[0])
        try:
            return solve_dense_cache[indices]
        except KeyError:
            x = solve_dense(A[:, S], p)
            solve_dense_cache[indices] = x
            return x

    for eta in etas:
        xs_ = warm.iterate(A=A, p=p, D=D, eta=eta, is_kl_not_js=False, q=None)

        for x in itertools.islice((x for i, x in enumerate(xs_) if i in I), len(I)):
            S = numpy.full(n, False)
            S[sorting.argmaxs(x, k)] = True
            x = numpy.zeros(n)
            x[S] = cached_solve_dense(S)

            q = A[:, S] @ x[S]
            divergence = D(p, q)
            if divergence < best_divergence:
                best_x = x
                best_divergence = divergence

            for l in L:
                r = p - q
                r_normalized = r / numpy.sum(numpy.abs(r))
                shift = A - q[:, None]
                shift_normalized = shift / numpy.sum(numpy.abs(shift), axis=0)
                divergences = D(r_normalized, shift_normalized)

                S[sorting.argmins(divergences, l)] = True
                x = numpy.zeros(n)
                x[S] = cached_solve_dense(S)

                S.fill(False)
                S[sorting.argmaxs(x, k)] = True
                x = numpy.zeros(n)
                x[S] = cached_solve_dense(S)

                q = A[:, S] @ x[S]
                divergence = D(p, q)
                if divergence < best_divergence:
                    best_x = x
                    best_divergence = divergence

    return best_x
