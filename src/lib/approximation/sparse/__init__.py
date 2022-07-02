"""
Approx. minimizes D(p, Cy) for y ∈ ℝ₊ⁿ subject to ‖y‖₀ ≤ k where cᵢ, p ∈ Δᵐ⁻¹.

This is a case of sparse approximation.

Δᵐ denotes the standard m-simplex, that is, the set of m+1-dimensional vectors
with nonnegative entries that sum up to 1 (probability vectors).

We have
    C ∈ ℝᵐˣⁿ with each column cᵢ ∈ Δᵐ⁻¹,
    p ∈ Δᵐ⁻¹,
    D is a divergence, and
    1 ≤ k < n.
"""
