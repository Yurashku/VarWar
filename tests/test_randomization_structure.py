from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec
from varwar.randomization_structure import PrognosticScoreBlockBuilder


def _history(n_rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(72)
    x = rng.normal(size=n_rows)
    category = np.resize(np.array(["a", "b", "c"], dtype=object), n_rows)
    y_pre = 1.5 * x + (category == "c") * 0.8 + rng.normal(scale=0.25, size=n_rows)
    return pd.DataFrame(
        {
            "unit": [f"u-{index}" for index in range(n_rows)],
            "x": x,
            "category": category,
            "y_pre": y_pre,
            "y": rng.normal(size=n_rows),
            "treatment": rng.integers(0, 2, size=n_rows),
        },
        index=pd.Index(np.arange(1000, 1000 + n_rows), name="row"),
    )


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        unit_col="unit",
        target_col="y",
        treatment_col="treatment",
        pre_target_col="y_pre",
        pre_treatment_cols=("y_pre", "x", "category"),
    )


class PrognosticScoreBlockBuilderTests(unittest.TestCase):
    def test_fit_transform_is_reproducible_and_categorical(self) -> None:
        data = _history()
        first = PrognosticScoreBlockBuilder(n_blocks=6)
        second = PrognosticScoreBlockBuilder(n_blocks=6)
        labels_a = first.fit_transform(data, _spec())
        labels_b = second.fit_transform(data, _spec())

        pd.testing.assert_series_equal(labels_a, labels_b)
        self.assertIsInstance(labels_a.dtype, pd.CategoricalDtype)
        self.assertTrue(labels_a.index.equals(data.index))
        self.assertLessEqual(labels_a.nunique(), 6)
        self.assertGreater(labels_a.nunique(), 1)

    def test_experiment_outcome_and_treatment_do_not_change_blocks(self) -> None:
        history = _history()
        builder = PrognosticScoreBlockBuilder(n_blocks=5).fit(history, _spec())
        experiment = history.copy()
        changed = history.copy()
        changed["y"] = np.arange(len(changed), dtype=float) ** 3
        changed["treatment"] = 1 - changed["treatment"]

        pd.testing.assert_series_equal(
            builder.transform(experiment), builder.transform(changed)
        )

    def test_unknown_categories_use_frozen_encoding(self) -> None:
        history = _history()
        builder = PrognosticScoreBlockBuilder(n_blocks=5).fit(history, _spec())
        future = history.iloc[:4].copy()
        future.loc[future.index[0], "category"] = "new-level"
        labels = builder.transform(future)

        self.assertFalse(labels.isna().any())
        self.assertEqual(builder.diagnostics()["unknown_category_rows"], 1)

    def test_augment_does_not_mutate_input(self) -> None:
        data = _history()
        builder = PrognosticScoreBlockBuilder(n_blocks=4).fit(data, _spec())
        augmented = builder.augment(data)
        self.assertNotIn("prognostic_block", data.columns)
        self.assertIn("prognostic_block", augmented.columns)

    def test_rejects_experiment_outcome_as_training_target(self) -> None:
        data = _history()
        builder = PrognosticScoreBlockBuilder(reference_target_col="y")
        with self.assertRaisesRegex(ValueError, "experiment outcome"):
            builder.fit(data, _spec())

    def test_rejects_non_pre_treatment_predictor(self) -> None:
        data = _history()
        builder = PrognosticScoreBlockBuilder(feature_cols=("x", "y"))
        with self.assertRaisesRegex(ValueError, "pre-treatment"):
            builder.fit(data, _spec())

    def test_constant_score_collapses_boundaries(self) -> None:
        data = _history(30)
        data["x"] = 1.0
        data["category"] = "a"
        data["y_pre"] = 2.0
        builder = PrognosticScoreBlockBuilder(n_blocks=7).fit(data, _spec())
        labels = builder.transform(data)
        self.assertEqual(labels.nunique(), 1)
        self.assertEqual(builder.diagnostics()["actual_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
