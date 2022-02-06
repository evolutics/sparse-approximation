"""
Minimizes D(b, Ax) for x ∈ ℝ₊ⁿ subject to ‖x‖₀ ≤ k where aₙ, b ∈ Δᵐ.

This is a case of sparse approximation.

Δ^L denotes the standard L-simplex, that is, the set of L-dimensional vectors
with nonnegative entries that sum up to 1 (probability vectors).

We have
    A ∈ ℝᵐˣⁿ with each column aₙ ∈ Δᵐ,
    b ∈ Δᵐ,
    D is a divergence, and
    1 ≤ k < n.
"""
