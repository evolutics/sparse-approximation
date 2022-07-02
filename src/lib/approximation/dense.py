"""
Minimizes D(p, Cy) for y ∈ Δⁿ⁻¹ where cᵢ, p ∈ ℝ₊ᵐ and D is a divergence.

These occur as ingredients of algorithms for the sparse case.
"""


import math

import cvxpy
import numpy


def euclidean(C, p):
    return _solve_convex(C, p, lambda p, q: cvxpy.norm2(p - q))


def total_variation(C, p):
    return _solve_convex(C, p, lambda p, q: 0.5 * cvxpy.norm1(p - q))


def _solve_convex(C, p, D):
    y = cvxpy.Variable(C.shape[1])
    objective = cvxpy.Minimize(D(p, C @ y))
    constraints = [y >= 0, cvxpy.sum(y) == 1]
    problem = cvxpy.Problem(objective, constraints)

    problem.solve()

    status = problem.status
    assert status == cvxpy.OPTIMAL, f"Unable to solve optimization problem: {status}"
    y = y.value

    assert math.isclose(numpy.sum(y), 1)
    y[numpy.isclose(y, 0)] = 0
    y /= numpy.sum(y)
    assert math.isclose(numpy.sum(y), 1)

    return y
