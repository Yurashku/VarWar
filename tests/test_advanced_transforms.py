from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from varwar.advanced_transforms import (
    AutoMLTemporalCupacTransformer,
    RobustHistoricalCupedTransformer,
)
from varwar.core import ExperimentSpec


def _temporal_data(n: int = 240, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x_lag2 = rng.normal(size=n)
    z_lag2 = rng.normal(size=n)
    y_pre = 0.5 + 1.8 * x_lag2 - 0.7 * z_lag2 + rng.normal(0.0, 0.2, n)
    x_lag1 = x_lag2 + rng.normal(0.0, 0.15, n)
    z_lag1 = z_lag2 + rng.normal(0.0, 0.15, n)
    y = 2.0 * y_pre + rng.normal(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "treatment": np.arange(n) % 2,
            "y": y,
            "y_pre": y_pre,
            "x": x_lag1,
            "z": z_lag1,
            "y_pre__lag2": 0.4 + 1.7 * x_lag2 - 0.6 * z_lag2,
            "x__lag2": x_lag2,
            "z__lag2": z_lag2,
        }
    )


class AutoMLTemporalCupacTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ExperimentSpec(
            pre_target_col="y_pre",
            pre_treatment_cols=("y_pre", "x", "z"),
        )

    def test_current_outcome_cannot_change_fit_or_model_selection(self) -> None:
        data = _temporal_data()
        changed = data.copy()
        changed["y"] = np.linspace(-10000.0, 10000.0, len(changed))
        settings = {
            "candidate_ids": ("ridge", "random_forest"),
            "cv_splits": 3,
            "random_state": 9,
            "n_estimators": 12,
        }
        first = AutoMLTemporalCupacTransformer(**settings)
        second = AutoMLTemporalCupacTransformer(**settings)

        adjusted_first = first.fit_transform(data, self.spec)
        adjusted_second = second.fit_transform(changed, self.spec)

        self.assertEqual(
            first.diagnostics()["selected_model"],
            second.diagnostics()["selected_model"],
        )
        self.assertEqual(first.diagnostics()["cv_mse"], second.diagnostics()["cv_mse"])
        np.testing.assert_allclose(
            adjusted_second.to_numpy() - adjusted_first.to_numpy(),
            changed["y"].to_numpy() - data["y"].to_numpy(),
        )
        self.assertFalse(first.diagnostics()["current_outcome_used_for_model_fit"])

    def test_same_seed_reproduces_selection_scores_and_adjustment(self) -> None:
        data = _temporal_data(seed=31)
        settings = {
            "candidate_ids": ("ridge", "extra_trees"),
            "cv_splits": 3,
            "random_state": 22,
            "n_estimators": 12,
        }
        first = AutoMLTemporalCupacTransformer(**settings)
        second = AutoMLTemporalCupacTransformer(**settings)
        first_values = first.fit_transform(data, self.spec)
        second_values = second.fit_transform(data, self.spec)

        self.assertEqual(first.diagnostics()["selected_model"], second.diagnostics()["selected_model"])
        self.assertEqual(first.diagnostics()["cv_mse"], second.diagnostics()["cv_mse"])
        np.testing.assert_allclose(first_values, second_values, rtol=0.0, atol=0.0)

    def test_model_selection_evaluates_every_declared_candidate(self) -> None:
        rng = np.random.default_rng(5)
        n = 240
        x2 = rng.uniform(-2.0, 2.0, n)
        y_pre = np.where(x2 > 0.25, 3.0, -2.0) + rng.normal(0.0, 0.08, n)
        data = pd.DataFrame(
            {
                "treatment": np.arange(n) % 2,
                "y": y_pre + rng.normal(0.0, 0.5, n),
                "y_pre": y_pre,
                "x": x2,
                "y_pre__lag2": np.where(x2 > 0.25, 2.8, -1.8),
                "x__lag2": x2,
            }
        )
        spec = ExperimentSpec(pre_target_col="y_pre", pre_treatment_cols=("x",))
        transformer = AutoMLTemporalCupacTransformer(
            random_state=4,
            cv_splits=3,
            n_estimators=16,
            max_tree_depth=5,
            hist_max_iter=40,
        )
        transformer.fit_transform(data, spec)
        diagnostics = transformer.diagnostics()

        self.assertEqual(set(diagnostics["cv_mse"]), set(transformer.candidate_ids))
        self.assertEqual(
            diagnostics["selected_model"],
            min(transformer.candidate_ids, key=lambda name: diagnostics["cv_mse"][name]),
        )
        self.assertIn(
            diagnostics["selected_model"],
            {"quadratic_ridge", "random_forest", "extra_trees", "hist_gradient_boosting"},
        )

    def test_fit_and_transform_are_reusable(self) -> None:
        data = _temporal_data()
        fit_frame = data.drop(columns="y")
        transformer = AutoMLTemporalCupacTransformer(
            candidate_ids=("ridge",), random_state=1
        ).fit(fit_frame, self.spec)
        first = transformer.transform(data.iloc[:100], self.spec)
        second = transformer.transform(data.iloc[100:], self.spec)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(second), len(data) - 100)


class RobustHistoricalCupedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ExperimentSpec(pre_target_col="y_pre")

    def test_huber_fit_never_reads_current_outcome(self) -> None:
        data = _temporal_data()
        changed = data.copy()
        changed["y"] = changed["y"] ** 3 + 500.0
        first = RobustHistoricalCupedTransformer()
        second = RobustHistoricalCupedTransformer()
        first_values = first.fit_transform(data, self.spec)
        second_values = second.fit_transform(changed, self.spec)

        self.assertEqual(first.diagnostics()["theta"], second.diagnostics()["theta"])
        np.testing.assert_allclose(
            second_values.to_numpy() - first_values.to_numpy(),
            changed["y"].to_numpy() - data["y"].to_numpy(),
        )
        self.assertFalse(first.diagnostics()["current_outcome_used_for_model_fit"])

    def test_huber_slope_is_robust_to_historical_outlier(self) -> None:
        rng = np.random.default_rng(13)
        x = np.linspace(-3.0, 3.0, 120)
        y_pre = 2.0 * x + rng.normal(0.0, 0.05, len(x))
        y_pre[0] = 500.0
        data = pd.DataFrame(
            {
                "y": rng.normal(size=len(x)),
                "y_pre": y_pre,
                "y_pre__lag2": x,
            }
        )
        transformer = RobustHistoricalCupedTransformer().fit(data.drop(columns="y"), self.spec)
        self.assertAlmostEqual(transformer.diagnostics()["theta"], 2.0, delta=0.12)
        self.assertGreater(transformer.diagnostics()["huber_outlier_fraction"], 0.0)

    def test_fit_is_reproducible_and_transform_preserves_index(self) -> None:
        data = _temporal_data().set_index(pd.Index(np.arange(1000, 1240)))
        first = RobustHistoricalCupedTransformer()
        second = RobustHistoricalCupedTransformer()
        first_values = first.fit_transform(data, self.spec)
        second_values = second.fit_transform(data, self.spec)
        self.assertEqual(first.diagnostics(), second.diagnostics())
        np.testing.assert_allclose(first_values, second_values, rtol=0.0, atol=0.0)
        self.assertTrue(first_values.index.equals(data.index))


if __name__ == "__main__":
    unittest.main()
