"""
Minimizes D(p, Cx) for x ∈ ℝ₊ⁿ where cᵢ, p ∈ Δᵐ and D is a divergence.

These occur as ingredients of algorithms for the sparse case.
"""


import cvxpy
import numpy


def euclidean(C, p):
    return _solve_convex(C, p, lambda p, q: cvxpy.norm2(p - q))


def total_variation(C, p):
    return _solve_convex(C, p, lambda p, q: 0.5 * cvxpy.norm1(p - q))


def _solve_convex(C, p, D):
    x = cvxpy.Variable(C.shape[1])
    objective = cvxpy.Minimize(D(p, C @ x))
    constraints = [x >= 0]
    problem = cvxpy.Problem(objective, constraints)

    problem.solve()

    status = problem.status
    assert status == cvxpy.OPTIMAL, f"Unable to solve optimization problem: {status}"
    x = x.value

    x[numpy.isclose(x, 0)] = 0
    return x
