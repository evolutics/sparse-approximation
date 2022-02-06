"""
Minimizes D(b, Ax) for x ∈ ℝ₊^N where aₙ, b ∈ Δ^M and D is a divergence.

These occur as ingredients of algorithms for the sparse case.
"""


import cvxpy
import numpy


def euclidean(A, b):
    return _solve_convex(A, b, lambda p, q: cvxpy.norm2(p - q))


def total_variation(A, b):
    return _solve_convex(A, b, lambda p, q: 0.5 * cvxpy.norm1(p - q))


def _solve_convex(A, b, D):
    x = cvxpy.Variable(A.shape[1])
    objective = cvxpy.Minimize(D(b, A @ x))
    constraints = [x >= 0]
    problem = cvxpy.Problem(objective, constraints)

    problem.solve()

    status = problem.status
    assert status == cvxpy.OPTIMAL, f"Unable to solve optimization problem: {status}"
    x = x.value

    x[numpy.isclose(x, 0)] = 0
    return x
