import itertools

import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm


def solve(C, p, D, k, *, solve_dense, eta, I):
    n = C.shape[1]

    xs_ = warm.iterate(C=C, p=p, D=D, eta=eta, is_kl_not_js=False, q=None)
    x = next(itertools.islice(xs_, I, None))

    S = numpy.full(n, False)
    S[sorting.argmaxs(x, k)] = True

    x = numpy.zeros(n)
    x[S] = solve_dense(C[:, S], p)

    return x
