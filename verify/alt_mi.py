"""Alternative Mutual Information estimator using histogram binning.

M4 cross-tool verification: this uses np.histogram2d + entropy formula,
which is a fundamentally different algorithm from the KSG k-NN estimator
used in src/discovery/mi_screening.py (via sklearn).

Both should agree on:
- MI(X, X) >> 0 (high self-information)
- MI(independent) ~ 0
- MI(correlated) > MI(independent)
- Ranking of pairs by MI should be similar

Exact values WILL differ (histogram vs KNN are different estimators),
so we compare rankings and relative magnitudes, not exact values.
"""

from __future__ import annotations

import numpy as np
from scipy.special import rel_entr


def compute_mi_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 20,
) -> float:
    """Compute MI via 2D histogram (plugin estimator).

    MI(X,Y) = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))

    This is the simplest MI estimator. It underestimates MI for
    continuous variables but preserves ranking for moderate sample sizes.

    Parameters
    ----------
    x, y : 1D arrays of same length
    bins : number of histogram bins per dimension

    Returns
    -------
    MI in nats (natural log), clipped to >= 0.
    """
    assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    assert len(x) >= 20, f"Too few observations: {len(x)}"

    # Joint histogram -> joint probability
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = hist_2d / hist_2d.sum()

    # Marginals
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # MI = KL(p_xy || p_x * p_y)
    p_xy_flat = p_xy.ravel()
    p_independent = np.outer(p_x, p_y).ravel()

    # rel_entr handles 0*log(0) = 0 correctly
    mi = np.sum(rel_entr(p_xy_flat, p_independent))

    return max(float(mi), 0.0)


def compute_mi_with_pvalue_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 20,
    n_permutations: int = 200,
    threshold: float = 0.05,
) -> dict:
    """MI with permutation-based p-value (histogram estimator).

    Same permutation logic as src/ but different MI estimator.
    """
    mi_observed = compute_mi_histogram(x, y, bins=bins)

    rng = np.random.default_rng(42)
    count_ge = 0

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        mi_perm = compute_mi_histogram(x, y_perm, bins=bins)
        if mi_perm >= mi_observed:
            count_ge += 1

        # Early stopping (same logic as src/)
        if i >= 9:
            best_possible = (count_ge + 1) / (n_permutations + 1)
            if best_possible >= threshold:
                break

    pvalue = (count_ge + 1) / (n_permutations + 1)
    return {
        "mi": mi_observed,
        "pvalue": pvalue,
        "significant": pvalue < threshold,
        "method": "histogram",
    }
