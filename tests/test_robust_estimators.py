from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec
from varwar.robust_estimators import (
    CatoniMeanEstimator,
    FullRefitWinsorizedMeanEstimator,
    IsolationForestSensitivityEstimator,
    MedianOfMeansEstimator,
    YuenTrimmedMeanEstimator,
)


def _frame(seed: int = 7, n: int = 160, effect: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = np.repeat([0, 1], n // 2)
    outcome = rng.normal(size=n) + effect * treatment
    outcome[[3, 91]] = [-15.0, 18.0]
    return pd.DataFrame({"treatment": treatment, "y": outcome})


class RobustEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = _frame()
        self.spec = ExperimentSpec()

    def test_yuen_matches_direct_trimmed_difference(self) -> None:
        estimator = YuenTrimmedMeanEstimator(trim_fraction=0.1)
        result = estimator.estimate(self.data, self.spec)
        control = np.sort(self.data.loc[self.data.treatment == 0, "y"].to_numpy())
        treatment = np.sort(self.data.loc[self.data.treatment == 1, "y"].to_numpy())
        removed = int(0.1 * control.size)
        expected = treatment[removed:-removed].mean() - control[removed:-removed].mean()
        self.assertAlmostEqual(result.effect, expected)
        self.assertEqual(result.diagnostics["estimand_id"], "trimmed_mean_effect")
        self.assertEqual(result.diagnostics["theory_grade"], "A")

    def test_winsorized_bootstrap_is_reproducible_and_refits_bounds(self) -> None:
        kwargs = dict(n_resamples=119, random_state=42, lower_quantile=0.05, upper_quantile=0.95)
        first = FullRefitWinsorizedMeanEstimator(**kwargs).estimate(self.data, self.spec)
        second = FullRefitWinsorizedMeanEstimator(**kwargs).estimate(self.data, self.spec)
        self.assertEqual(first.to_record(), second.to_record())
        self.assertGreater(first.standard_error, 0.0)
        self.assertEqual(first.diagnostics["valid_resamples"], 119)
        self.assertEqual(first.diagnostics["ate_alignment"], "not_ate")

    def test_median_of_means_targets_original_mean_and_is_reproducible(self) -> None:
        estimator = MedianOfMeansEstimator(n_blocks=5, n_resamples=119, random_state=19)
        first = estimator.estimate(self.data, self.spec)
        second = estimator.estimate(self.data, self.spec)
        self.assertEqual(first.to_record(), second.to_record())
        self.assertEqual(first.diagnostics["estimand_id"], "sample_ate")
        self.assertEqual(first.diagnostics["theory_grade"], "C")
        self.assertTrue(first.diagnostics["conservative_envelope"])
        self.assertTrue(first.ci_low <= first.ci_high)

    def test_catoni_is_location_equivariant(self) -> None:
        estimator = CatoniMeanEstimator(n_resamples=119, random_state=23)
        original = estimator.estimate(self.data, self.spec)
        shifted = self.data.copy()
        shifted["y"] += 1000.0
        translated = estimator.estimate(shifted, self.spec)
        self.assertAlmostEqual(original.effect, translated.effect, places=9)
        self.assertAlmostEqual(original.standard_error, translated.standard_error, places=9)
        self.assertEqual(original.diagnostics["estimand_id"], "sample_ate")
        self.assertTrue(original.diagnostics["conservative_envelope"])
        self.assertIn("variance bound", original.diagnostics["inference_limitation"])

    def test_invalid_resample_count_is_rejected(self) -> None:
        estimator = FullRefitWinsorizedMeanEstimator(n_resamples=20)
        with self.assertRaisesRegex(ValueError, "at least 99"):
            estimator.estimate(self.data, self.spec)

    def test_non_finite_targets_are_removed_per_arm(self) -> None:
        data = self.data.copy()
        data.loc[[0, 81], "y"] = [np.nan, np.inf]
        result = YuenTrimmedMeanEstimator().estimate(data, self.spec)
        self.assertEqual(result.n_control, 79)
        self.assertEqual(result.n_treatment, 79)

    def test_isolation_forest_optional_dependency_error_is_clear(self) -> None:
        estimator = IsolationForestSensitivityEstimator(n_resamples=99)
        real_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ImportError, "'ml' extra"):
                estimator.estimate(self.data, self.spec)

    def test_isolation_forest_refit_smoke_when_available(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        estimator = IsolationForestSensitivityEstimator(
            method="winsorize",
            contamination=0.05,
            n_estimators=32,
            n_resamples=99,
            random_state=31,
        )
        result = estimator.estimate(self.data, self.spec)
        self.assertTrue(np.isfinite(result.effect))
        self.assertTrue(result.diagnostics["full_refit"])
        self.assertEqual(result.diagnostics["valid_resamples"], 99)
        self.assertEqual(result.diagnostics["ate_alignment"], "not_ate")


if __name__ == "__main__":
    unittest.main()
