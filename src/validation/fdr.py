"""Benjamini-Hochberg False Discovery Rate correction.

Controls the expected proportion of false positives among rejected hypotheses.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(
    pvalues: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> list[bool]:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        pvalues: Array of p-values.
        alpha: Target FDR level.

    Returns:
        List of booleans — True if the corresponding hypothesis is significant
        after FDR correction.
    """
    pvals = np.asarray(pvalues, dtype=float)
    n = len(pvals)
    if n == 0:
        return []

    # Sort p-values and track original indices
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]

    # BH critical values: (rank / n) * alpha
    ranks = np.arange(1, n + 1)
    thresholds = (ranks / n) * alpha

    # Find largest k where p_(k) <= threshold_(k)
    significant = np.zeros(n, dtype=bool)
    # All p-values at rank <= k_max are rejected
    rejections = sorted_pvals <= thresholds
    if rejections.any():
        k_max = np.max(np.where(rejections)[0])
        significant[sorted_indices[: k_max + 1]] = True

    return significant.tolist()
