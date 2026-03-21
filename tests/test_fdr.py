"""Tests for FDR correction (ST-08).

L1 tests verify Benjamini-Hochberg implementation on synthetic p-values.
"""

from __future__ import annotations

import numpy as np

from src.validation.fdr import benjamini_hochberg


class TestFDR:
    """L1: Benjamini-Hochberg FDR correction."""

    def test_all_significant(self) -> None:
        """All very small p-values should be significant."""
        pvalues = [0.001, 0.002, 0.003, 0.004, 0.005]
        result = benjamini_hochberg(pvalues, alpha=0.05)
        assert all(result)

    def test_all_nonsignificant(self) -> None:
        """All large p-values should be non-significant."""
        pvalues = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = benjamini_hochberg(pvalues, alpha=0.05)
        assert not any(result)

    def test_mixed_pvalues(self) -> None:
        """Synthetic mix: 950 null + 50 real signals.

        SOURCE: Benjamini & Hochberg (1995), "Controlling the FDR"
        With 950 U(0,1) + 50 Beta(0.1,1) p-values at α=0.05,
        we expect ~50 rejections with FDR controlled at 5%.
        """
        rng = np.random.default_rng(42)
        null_pvals = rng.uniform(0, 1, 950)
        # Beta(0.1, 1) concentrates near 0 — simulates real signals
        signal_pvals = rng.beta(0.1, 1, 50)
        all_pvals = np.concatenate([null_pvals, signal_pvals])

        result = benjamini_hochberg(all_pvals.tolist(), alpha=0.05)
        n_rejected = sum(result)

        # Should find most of the 50 signals
        assert n_rejected >= 20, f"Too few rejections: {n_rejected}"
        # FDR should be controlled — not too many false positives
        assert n_rejected < 150, f"Too many rejections: {n_rejected}"

    def test_empty_input(self) -> None:
        result = benjamini_hochberg([], alpha=0.05)
        assert result == []

    def test_single_significant(self) -> None:
        result = benjamini_hochberg([0.01], alpha=0.05)
        assert result == [True]

    def test_single_nonsignificant(self) -> None:
        result = benjamini_hochberg([0.1], alpha=0.05)
        assert result == [False]

    def test_preserves_order(self) -> None:
        """Results should match input order, not sorted order."""
        pvalues = [0.5, 0.001, 0.8, 0.002, 0.9]
        result = benjamini_hochberg(pvalues, alpha=0.05)
        assert result[1] is True  # 0.001
        assert result[3] is True  # 0.002
        assert result[0] is False  # 0.5
