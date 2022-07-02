"""
Approx. minimizes D(p, Cy) for y ∈ Δⁿ⁻¹ subject to ‖y‖₀ ≤ k where cᵢ, p ∈ ℝ₊ᵐ.

This is a case of sparse approximation.

Δᵐ denotes the standard m-simplex, that is, the set of m+1-dimensional vectors
with nonnegative entries that sum up to 1 (probability vectors).

We have
    C ∈ ℝ₊ᵐˣⁿ,
    p ∈ ℝ₊ᵐ,
    D is a generalized "divergence", and
    1 ≤ k < n.
"""
