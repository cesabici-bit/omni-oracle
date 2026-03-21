"""Mutual Information screening with permutation-based p-values.

Uses sklearn's KSG estimator (Kraskov et al. 2004) for MI estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from src.config import MI_PERMUTATIONS, MI_PVALUE_THRESHOLD


@dataclass
class MIResult:
    """Result of MI computation between two variables."""

    mi: float  # MI in nats
    pvalue: float  # permutation p-value
    significant: bool  # pvalue < threshold


def compute_mi(
    x: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 7,
) -> float:
    """Compute Mutual Information between x and y using KSG estimator.

    Args:
        x: 1D array of observations.
        y: 1D array of observations (same length as x).
        n_neighbors: Number of neighbors for KSG estimator.

    Returns:
        MI estimate in nats (non-negative by construction).
    """
    assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    assert len(x) >= 20, f"Too few observations for MI: {len(x)}"

    x_2d = x.reshape(-1, 1)
    # mutual_info_regression treats first arg as features, second as target
    mi_values = mutual_info_regression(
        x_2d, y, n_neighbors=n_neighbors, random_state=42
    )
    return float(max(mi_values[0], 0.0))  # MI >= 0 by definition


def compute_mi_with_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = MI_PERMUTATIONS,
    n_neighbors: int = 7,
    threshold: float = MI_PVALUE_THRESHOLD,
) -> MIResult:
    """Compute MI with permutation-based p-value.

    The p-value is the fraction of permuted MI values >= observed MI.
    Uses early stopping: if after k permutations the p-value cannot
    possibly become significant, we stop early to save computation.
    """
    observed_mi = compute_mi(x, y, n_neighbors=n_neighbors)

    rng = np.random.default_rng(42)
    count_ge = 0
    for k in range(1, n_permutations + 1):
        y_perm = rng.permutation(y)
        mi_perm = compute_mi(x, y_perm, n_neighbors=n_neighbors)
        if mi_perm >= observed_mi:
            count_ge += 1

        # Early stopping: if current p-value is already hopeless, bail out.
        # Even if no more permutations exceed observed_mi, pvalue would be
        # (count_ge + 1) / (n_permutations + 1). If that's already >= threshold, stop.
        if k >= 10:  # wait at least 10 permutations
            best_possible_pvalue = (count_ge + 1) / (n_permutations + 1)
            if best_possible_pvalue >= threshold:
                pvalue = (count_ge + 1) / (k + 1)
                return MIResult(mi=observed_mi, pvalue=pvalue, significant=False)

    pvalue = (count_ge + 1) / (n_permutations + 1)  # +1 for conservative estimate

    return MIResult(
        mi=observed_mi,
        pvalue=pvalue,
        significant=pvalue < threshold,
    )


def screen_all_pairs(
    variables: dict[str, np.ndarray],
    n_permutations: int = MI_PERMUTATIONS,
    threshold: float = MI_PVALUE_THRESHOLD,
) -> list[tuple[str, str, MIResult]]:
    """Screen all pairs of variables for significant MI.

    Args:
        variables: Dict mapping variable_id → 1D numpy array (aligned).
        n_permutations: Number of permutations for p-value.
        threshold: p-value threshold.

    Returns:
        List of (var_x, var_y, MIResult) for significant pairs only.
    """
    var_ids = sorted(variables.keys())
    significant_pairs: list[tuple[str, str, MIResult]] = []

    for i in range(len(var_ids)):
        for j in range(i + 1, len(var_ids)):
            x = variables[var_ids[i]]
            y = variables[var_ids[j]]

            # Skip if different lengths (not aligned)
            if len(x) != len(y):
                continue

            result = compute_mi_with_pvalue(
                x, y,
                n_permutations=n_permutations,
                threshold=threshold,
            )
            if result.significant:
                significant_pairs.append((var_ids[i], var_ids[j], result))

    return significant_pairs
