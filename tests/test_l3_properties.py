"""L3 Property-Based Tests — verify statistical invariants hold for any valid input.

Uses Hypothesis library for property-based testing.
These ensure that our implementations respect mathematical invariants
regardless of the specific input data.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.discovery.mi_screening import compute_mi
from src.validation.fdr import benjamini_hochberg

# --- Strategies ---

def float_arrays(min_size: int = 50, max_size: int = 300):
    """Generate valid float arrays for statistical tests."""
    return arrays(
        dtype=np.float64,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    )


def pvalue_lists(min_size: int = 1, max_size: int = 200):
    """Generate valid p-value lists."""
    return st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=min_size,
        max_size=max_size,
    )


class TestMIProperties:
    """L3: Mathematical invariants of Mutual Information."""

    @given(data=st.data())
    @settings(max_examples=20, deadline=30000)
    def test_mi_nonnegative(self, data: st.DataObject) -> None:
        """MI(X, Y) >= 0 for any valid input.

        SOURCE: Cover & Thomas (2006), Theorem 2.6.3 — MI is non-negative.
        """
        n = data.draw(st.integers(min_value=50, max_value=200))
        seed = data.draw(st.integers(min_value=0, max_value=10000))
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        mi = compute_mi(x, y)
        assert mi >= 0, f"MI must be non-negative, got {mi}"

    @given(data=st.data())
    @settings(max_examples=20, deadline=30000)
    def test_mi_symmetric(self, data: st.DataObject) -> None:
        """MI(X, Y) ≈ MI(Y, X) for any valid input.

        SOURCE: Cover & Thomas (2006), Theorem 2.4.1 — MI is symmetric.
        """
        n = data.draw(st.integers(min_value=50, max_value=200))
        seed = data.draw(st.integers(min_value=0, max_value=10000))
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        mi_xy = compute_mi(x, y)
        mi_yx = compute_mi(y, x)
        # KSG estimator has some randomness in tie-breaking, allow tolerance
        assert abs(mi_xy - mi_yx) < 0.15, (
            f"MI should be symmetric: MI(X,Y)={mi_xy:.4f} vs MI(Y,X)={mi_yx:.4f}"
        )


class TestFDRProperties:
    """L3: Mathematical invariants of Benjamini-Hochberg."""

    @given(pvalues=pvalue_lists(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_fdr_output_same_length(self, pvalues: list[float]) -> None:
        """BH output has same length as input."""
        result = benjamini_hochberg(pvalues, alpha=0.05)
        assert len(result) == len(pvalues)

    @given(pvalues=pvalue_lists(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_fdr_output_is_boolean(self, pvalues: list[float]) -> None:
        """BH output is list of booleans."""
        result = benjamini_hochberg(pvalues, alpha=0.05)
        assert all(isinstance(r, bool) for r in result)

    @given(pvalues=pvalue_lists(min_size=2, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_fdr_monotone_in_alpha(self, pvalues: list[float]) -> None:
        """More rejections with larger α.

        SOURCE: Benjamini & Hochberg (1995) — BH procedure is monotone in α.
        """
        r_strict = sum(benjamini_hochberg(pvalues, alpha=0.01))
        r_loose = sum(benjamini_hochberg(pvalues, alpha=0.10))
        assert r_strict <= r_loose, (
            f"α=0.01 should reject ≤ α=0.10: {r_strict} > {r_loose}"
        )

    @given(data=st.data())
    @settings(max_examples=30, deadline=5000)
    def test_fdr_smaller_pvalue_more_likely_rejected(self, data: st.DataObject) -> None:
        """If p_i < p_j and p_j is rejected, then p_i must also be rejected.

        SOURCE: Benjamini & Hochberg (1995) — step-up property.
        """
        n = data.draw(st.integers(min_value=5, max_value=50))
        seed = data.draw(st.integers(min_value=0, max_value=10000))
        rng = np.random.default_rng(seed)
        pvalues = rng.uniform(0, 1, n).tolist()

        result = benjamini_hochberg(pvalues, alpha=0.05)

        for i in range(n):
            for j in range(n):
                if pvalues[i] < pvalues[j] and result[j]:
                    assert result[i], (
                        f"p[{i}]={pvalues[i]:.4f} < p[{j}]={pvalues[j]:.4f} "
                        f"but p[{j}] rejected while p[{i}] not"
                    )
