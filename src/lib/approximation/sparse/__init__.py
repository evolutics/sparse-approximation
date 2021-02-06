"""
Minimizes D(b, Ax) for x ∈ Δ^N subject to ‖x‖₀ ≤ K where aₙ, b ∈ Δ^M.

This is a case of sparse approximation.

Δ^L denotes the standard L-simplex, that is, the set of L-dimensional vectors
with nonnegative entries that sum up to 1 (probability vectors).

We have
    A ∈ ℝ^{M×N} with each column aₙ ∈ Δ^M,
    b ∈ Δ^M,
    D is a divergence, and
    1 ≤ K < N.
"""
