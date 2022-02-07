import itertools

import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm


def solve(C, p, D, k, *, solve_dense, eta, I):
    n = C.shape[1]

    ys = warm.iterate(C=C, p=p, D=D, eta=eta, is_kl_not_js=False, q=None)
    y = next(itertools.islice(ys, I, None))

    S = numpy.full(n, False)
    S[sorting.argmaxs(y, k)] = True

    y = numpy.zeros(n)
    y[S] = solve_dense(C[:, S], p)

    return y
