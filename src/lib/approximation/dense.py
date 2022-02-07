"""
Minimizes D(p, Ax) for x ∈ ℝ₊ⁿ where aᵢ, p ∈ Δᵐ and D is a divergence.

These occur as ingredients of algorithms for the sparse case.
"""


import cvxpy
import numpy


def euclidean(A, p):
    return _solve_convex(A, p, lambda p, q: cvxpy.norm2(p - q))


def total_variation(A, p):
    return _solve_convex(A, p, lambda p, q: 0.5 * cvxpy.norm1(p - q))


def _solve_convex(A, p, D):
    x = cvxpy.Variable(A.shape[1])
    objective = cvxpy.Minimize(D(p, A @ x))
    constraints = [x >= 0]
    problem = cvxpy.Problem(objective, constraints)

    problem.solve()

    status = problem.status
    assert status == cvxpy.OPTIMAL, f"Unable to solve optimization problem: {status}"
    x = x.value

    x[numpy.isclose(x, 0)] = 0
    return x
