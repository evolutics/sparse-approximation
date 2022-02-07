"""
Minimizes D(p, Ax) for x ∈ ℝ₊ⁿ subject to ‖x‖₀ ≤ k where aᵢ, p ∈ Δᵐ.

This is a case of sparse approximation.

Δ^L denotes the standard L-simplex, that is, the set of L-dimensional vectors
with nonnegative entries that sum up to 1 (probability vectors).

We have
    A ∈ ℝᵐˣⁿ with each column aᵢ ∈ Δᵐ,
    p ∈ Δᵐ,
    D is a divergence, and
    1 ≤ k < n.
"""
