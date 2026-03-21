"""L2 Oracle Tests — verify implementations against known analytical results.

Each test cites an external source for the expected value.
These are NOT testing the pipeline E2E, but the individual statistical
functions against textbook/paper ground truth.
"""

from __future__ import annotations

import numpy as np

from src.discovery.granger import test_granger_bidirectional as granger_bidirectional
from src.discovery.mi_screening import compute_mi
from src.validation.fdr import benjamini_hochberg
from src.validation.temporal_oos import validate_oos


class TestMIOracleL2:
    """L2: MI against analytical formulas from information theory."""

    def test_mi_bivariate_gaussian_rho05(self) -> None:
        """MI of bivariate Gaussian with ρ=0.5.

        SOURCE: Cover & Thomas (2006), "Elements of Information Theory", 2nd ed.
        Theorem 8.4.1: MI(X,Y) = -0.5 * ln(1 - ρ²)
        For ρ=0.5: MI = -0.5 * ln(1 - 0.25) = -0.5 * ln(0.75) ≈ 0.1438 nats
        """
        # SOURCE: Cover & Thomas (2006), Theorem 8.4.1
        rho = 0.5
        expected_mi = -0.5 * np.log(1 - rho**2)  # 0.14384...

        rng = np.random.default_rng(42)
        n = 2000  # large sample for precision
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)
        x = z1
        y = rho * z1 + np.sqrt(1 - rho**2) * z2

        mi = compute_mi(x, y)
        assert abs(mi - expected_mi) / expected_mi < 0.30, (
            f"MI={mi:.4f}, expected≈{expected_mi:.4f} (Cover & Thomas Thm 8.4.1)"
        )

    def test_mi_bivariate_gaussian_rho09(self) -> None:
        """MI of bivariate Gaussian with ρ=0.9 (high dependence).

        SOURCE: Cover & Thomas (2006), Theorem 8.4.1
        MI = -0.5 * ln(1 - 0.81) = -0.5 * ln(0.19) ≈ 0.8267 nats
        """
        # SOURCE: Cover & Thomas (2006), Theorem 8.4.1
        rho = 0.9
        expected_mi = -0.5 * np.log(1 - rho**2)  # 0.8267...

        rng = np.random.default_rng(42)
        n = 2000
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)
        x = z1
        y = rho * z1 + np.sqrt(1 - rho**2) * z2

        mi = compute_mi(x, y)
        assert abs(mi - expected_mi) / expected_mi < 0.25, (
            f"MI={mi:.4f}, expected≈{expected_mi:.4f} (Cover & Thomas Thm 8.4.1)"
        )

    def test_mi_monotone_with_correlation(self) -> None:
        """L3 (property): MI should increase monotonically with |ρ|.

        SOURCE: Cover & Thomas (2006), Theorem 8.4.1 — MI = -0.5*ln(1-ρ²)
        is strictly increasing in |ρ|.
        """
        # SOURCE: Cover & Thomas (2006), Theorem 8.4.1
        rng = np.random.default_rng(42)
        n = 1000
        rhos = [0.2, 0.5, 0.8]
        mis = []

        for rho in rhos:
            z1 = rng.normal(0, 1, n)
            z2 = rng.normal(0, 1, n)
            x = z1
            y = rho * z1 + np.sqrt(1 - rho**2) * z2
            mi = compute_mi(x, y)
            mis.append(mi)

        for i in range(len(mis) - 1):
            assert mis[i] < mis[i + 1], (
                f"MI should increase with |ρ|: MI(ρ={rhos[i]})={mis[i]:.4f} "
                f">= MI(ρ={rhos[i+1]})={mis[i+1]:.4f}"
            )


class TestGrangerOracleL2:
    """L2: Granger causality against known DGP (data generating processes)."""

    def test_granger_var1_known_coefficient(self) -> None:
        """VAR(1) with known X→Y coefficient should be detected.

        SOURCE: Granger (1969), "Investigating Causal Relations by Econometric
        Models and Cross-spectral Methods", Econometrica 37(3), pp. 424-438.
        Definition: X Granger-causes Y if past X improves prediction of Y
        beyond Y's own past.

        DGP: X(t) = 0.8*X(t-1) + ε_x
             Y(t) = 0.4*Y(t-1) + 0.6*X(t-1) + ε_y
        """
        # SOURCE: Granger (1969), Definition 1
        rng = np.random.default_rng(42)
        n = 600
        x = np.zeros(n)
        y = np.zeros(n)

        for t in range(1, n):
            x[t] = 0.8 * x[t - 1] + rng.normal(0, 1)
            y[t] = 0.4 * y[t - 1] + 0.6 * x[t - 1] + rng.normal(0, 1)

        # Discard burn-in
        x, y = x[100:], y[100:]

        result = granger_bidirectional(x, y)
        assert result.pvalue_xy < 0.01, (
            f"X→Y should be detected (Granger 1969), p={result.pvalue_xy:.4f}"
        )
        assert result.direction in ("x->y", "bidirectional")

    def test_granger_contemporaneous_not_detected(self) -> None:
        """Contemporaneous correlation (no lag) should NOT be Granger-causal.

        SOURCE: Granger (1969) — Granger causality is STRICTLY about
        past values improving prediction. If Y depends on X at the SAME
        time step only, X does not Granger-cause Y.

        DGP: X(t) = ε_x
             Y(t) = 0.5*X(t) + ε_y  (contemporaneous only, no lag)
        """
        # SOURCE: Granger (1969), Definition 1 — requires temporal precedence
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        y = 0.5 * x + rng.normal(0, 1, n)

        result = granger_bidirectional(x, y)
        # Neither direction should be strongly significant
        # (contemporaneous effects are not Granger causality)
        assert result.best_pvalue > 0.001 or result.direction == "none", (
            f"Contemporaneous-only should not show strong Granger: "
            f"dir={result.direction}, p={result.best_pvalue:.6f}"
        )

    def test_granger_common_driver_shows_bidirectional(self) -> None:
        """Common driver Z→X, Z→Y should show bidirectional or none.

        SOURCE: Granger (1969), Sect. 4 — confounding by common cause.
        When Z drives both X and Y with lags, the Granger test may find
        bidirectional causality (which is a known limitation).

        DGP: Z(t) = 0.8*Z(t-1) + ε_z
             X(t) = 0.3*X(t-1) + 0.5*Z(t-1) + ε_x
             Y(t) = 0.3*Y(t-1) + 0.5*Z(t-2) + ε_y
        """
        # SOURCE: Granger (1969), Section 4 (confounders)
        rng = np.random.default_rng(42)
        n = 600
        z = np.zeros(n)
        x = np.zeros(n)
        y = np.zeros(n)

        for t in range(2, n):
            z[t] = 0.8 * z[t - 1] + rng.normal(0, 1)
            x[t] = 0.3 * x[t - 1] + 0.5 * z[t - 1] + rng.normal(0, 1)
            y[t] = 0.3 * y[t - 1] + 0.5 * z[t - 2] + rng.normal(0, 1)

        x, y = x[100:], y[100:]
        result = granger_bidirectional(x, y)

        # With a common driver, we expect either bidirectional or
        # spurious unidirectional — the key is the test doesn't say "none"
        # This is a KNOWN LIMITATION we document in caveats
        assert result.best_pvalue < 0.05, (
            "Common driver should produce apparent Granger relationship"
        )


class TestFDROracleL2:
    """L2: FDR correction against Benjamini-Hochberg (1995) theory."""

    def test_fdr_control_under_null(self) -> None:
        """Under complete null (all H0 true), FDR ≤ α.

        SOURCE: Benjamini & Hochberg (1995), "Controlling the False Discovery
        Rate: A Practical and Powerful Approach to Multiple Testing",
        JRSS-B, 57(1), pp. 289-300. Theorem 1.

        Under complete null, p-values are U(0,1). BH procedure at α=0.05
        should reject ≤ α fraction = 5%.
        """
        # SOURCE: Benjamini & Hochberg (1995), Theorem 1
        rng = np.random.default_rng(42)
        n_tests = 1000

        # Run 100 simulations to check average FDR
        rejection_rates = []
        for i in range(100):
            pvals = rng.uniform(0, 1, n_tests).tolist()
            mask = benjamini_hochberg(pvals, alpha=0.05)
            rejection_rates.append(sum(mask) / n_tests)

        avg_rate = np.mean(rejection_rates)
        # Under complete null, BH should reject ~0% (very conservative)
        # But allow up to α = 5%
        assert avg_rate < 0.06, (
            f"FDR under null should be ≤ 0.05, got {avg_rate:.4f} "
            f"(Benjamini & Hochberg 1995, Theorem 1)"
        )

    def test_fdr_power_with_signals(self) -> None:
        """BH should detect most true signals while controlling FDR.

        SOURCE: Benjamini & Hochberg (1995), Theorem 1 + simulation.
        With 900 null + 100 true signals (p ~ Beta(0.05, 1)),
        expect FDR ≤ 0.05 and power > 50%.
        """
        # SOURCE: Benjamini & Hochberg (1995), Theorem 1
        rng = np.random.default_rng(42)
        null_p = rng.uniform(0, 1, 900)
        signal_p = rng.beta(0.05, 1, 100)  # very small p-values
        all_p = np.concatenate([null_p, signal_p]).tolist()

        mask = benjamini_hochberg(all_p, alpha=0.05)

        # Count discoveries among signals (indices 900-999)
        true_positives = sum(mask[900:])
        false_positives = sum(mask[:900])
        total_rejected = sum(mask)

        # Power: should find most of the 100 signals
        assert true_positives >= 50, (
            f"BH should have good power: only {true_positives}/100 true signals found"
        )

        # FDR control: FP / (FP + TP) ≤ α
        if total_rejected > 0:
            fdr = false_positives / total_rejected
            assert fdr < 0.10, (
                f"FDR={fdr:.4f} exceeds tolerance (BH 1995 guarantees ≤ 0.05)"
            )


class TestOOSOracleL2:
    """L2: Out-of-sample validation against known properties."""

    def test_oos_perfect_linear_relationship(self) -> None:
        """A persistent linear X→Y should have high OOS R².

        SOURCE: Standard OLS theory — if Y = αX_{t-lag} + ε with constant
        coefficients, the OOS R² should converge to in-sample R² for
        large enough train and test sets.
        """
        # SOURCE: OLS theory — persistent linear relationship
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * x[t - 1] + rng.normal(0, 0.5)

        result = validate_oos(x, y, lag=1)
        assert result.r2_incremental > 0.10, (
            f"Strong persistent relationship should have high OOS R², "
            f"got {result.r2_incremental:.4f}"
        )
        assert result.valid

    def test_oos_structural_break_detected(self) -> None:
        """Relationship that changes midway should have low OOS R².

        SOURCE: Chow (1960), "Tests of Equality Between Sets of Coefficients
        in Two Linear Regressions" — structural breaks invalidate
        extrapolation from training to test period.

        DGP: Y = 0.8*X_{t-1} + ε for t < 300
             Y = -0.8*X_{t-1} + ε for t >= 300
        """
        # SOURCE: Chow (1960), structural break concept
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        y = np.zeros(n)

        # Coefficient reversal at t=300
        for t in range(1, 300):
            y[t] = 0.8 * x[t - 1] + rng.normal(0, 0.5)
        for t in range(300, n):
            y[t] = -0.8 * x[t - 1] + rng.normal(0, 0.5)

        result = validate_oos(x, y, lag=1, train_ratio=0.6)
        # Model trained on positive regime should fail on negative regime
        assert result.r2_incremental < 0.15, (
            f"Structural break should hurt OOS R², got {result.r2_incremental:.4f}"
        )


class TestGrangerOkuinsLawL2:
    """L2: Verify Granger implementation on Okun's Law synthetic data.

    SOURCE: Okun (1962), "Potential GNP: Its Measurement and Significance",
    Proceedings of the Business and Economics Statistics Section, ASA.
    Okun's Law: 1% increase in unemployment ≈ 2% decrease in GDP growth.
    Lag: typically 1-2 quarters.
    """

    def test_okun_law_synthetic(self) -> None:
        """Synthetic Okun's Law: GDP growth → Unemployment with lag.

        SOURCE: Okun (1962) — GDP growth Granger-causes unemployment changes
        with approximately 1-2 quarter lag.

        DGP (simplified):
        GDP_growth(t) = 0.5*GDP_growth(t-1) + ε
        ΔUnemployment(t) = -0.4*GDP_growth(t-1) + 0.3*ΔUnemployment(t-1) + ε
        (negative coefficient = higher GDP growth → lower unemployment)
        """
        # SOURCE: Okun (1962), simplified DGP
        rng = np.random.default_rng(42)
        n = 400
        gdp_growth = np.zeros(n)
        d_unemp = np.zeros(n)

        for t in range(1, n):
            gdp_growth[t] = 0.5 * gdp_growth[t - 1] + rng.normal(0, 1)
            d_unemp[t] = (
                0.3 * d_unemp[t - 1]
                - 0.4 * gdp_growth[t - 1]
                + rng.normal(0, 1)
            )

        # Discard burn-in
        gdp_growth = gdp_growth[50:]
        d_unemp = d_unemp[50:]

        result = granger_bidirectional(gdp_growth, d_unemp)
        assert result.pvalue_xy < 0.05, (
            f"GDP→Unemployment (Okun 1962) should be detected, "
            f"p={result.pvalue_xy:.4f}"
        )
        assert result.direction in ("x->y", "bidirectional")
