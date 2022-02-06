import itertools

import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm


def solve(A, b, D, k, *, solve_dense, eta, I):
    n = A.shape[1]

    xs_ = warm.iterate(A=A, b=b, D=D, eta=eta, is_kl_not_js=False, q=None)
    x = next(itertools.islice(xs_, I, None))

    S = numpy.full(n, False)
    S[sorting.argmaxs(x, k)] = True

    x = numpy.zeros(n)
    x[S] = solve_dense(A[:, S], b)

    return x
