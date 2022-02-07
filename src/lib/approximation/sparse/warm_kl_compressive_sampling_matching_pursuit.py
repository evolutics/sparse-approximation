import numpy

from src.lib import sorting
from src.lib.approximation.sparse import warm
from src.lib.approximation.sparse import warm_kl


def solve(A, p, D, k, *, solve_dense, eta, I, L):
    n = A.shape[1]

    best_x = warm_kl.solve(A, p, D, k, solve_dense=solve_dense, eta=eta, I=I)
    S = best_x != 0
    q = A[:, S] @ best_x[S]
    best_divergence = D(p, q)

    for l in L:
        xs_ = warm.iterate(A=A, p=p, D=D, eta=eta, is_kl_not_js=True, q=q)
        x = next(x for i, x in enumerate(xs_) if numpy.count_nonzero(x) >= l or i >= I)
        S |= x != 0

        x = numpy.zeros(n)
        x[S] = solve_dense(A[:, S], p)

        S.fill(False)
        S[sorting.argmaxs(x, k)] = True

        q = A[:, S] @ x[S]

    x = numpy.zeros(n)
    x[S] = solve_dense(A[:, S], p)
    divergence = D(p, A[:, S] @ x[S])
    if divergence < best_divergence:
        best_x = x

    return best_x
