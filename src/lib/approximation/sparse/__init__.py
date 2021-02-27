"""
Minimizes D(b, Ax) for x ∈ ℝ₊^N subject to ‖x‖₀ ≤ K where aₙ, b ∈ ℝ₊^M.

This is a case of sparse approximation.

ℝ₊ denotes the nonnegative real numbers.

We have
    A ∈ ℝ₊^{M×N},
    b ∈ ℝ₊^M,
    D is a divergence, and
    1 ≤ K < N.
"""
