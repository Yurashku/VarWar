from __future__ import annotations

import os
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar.core import ExperimentSpec
from varwar.panel_prior import PanelPriorSummary
from varwar.synthetic import SyntheticConfig, generate_synthetic_ab, generate_synthetic_panel
from varwar.transforms import PanelPriorCupacTransformer


class PanelPriorTests(unittest.TestCase):
    def test_summary_contains_sql_like_statistics_and_rejects_invalid_cutoff(self) -> None:
        panel = generate_synthetic_panel(SyntheticConfig(n=50, seed=11), n_months=4)
        summary = PanelPriorSummary.from_frame(
            panel,
            feature_cols=("y_pre__lag2", "x1__lag2"),
            target_col="y_pre",
            date_col="report_date",
            cutoff="2023-05-01",
            entity_col="entity_id",
            category_cols=("region",),
        )
        self.assertEqual(summary.n_rows, 200)
        self.assertEqual(summary.n_entities, 50)
        self.assertEqual(set(summary.monthly_stats), {"2023-01", "2023-02", "2023-03", "2023-04"})
        self.assertAlmostEqual(sum(summary.category_shares["region"].values()), 1.0)
        with self.assertRaises(ValueError):
            PanelPriorSummary.from_frame(
                panel,
                feature_cols=("y_pre__lag2",),
                target_col="y_pre",
                date_col="report_date",
                cutoff=None,
            )
        future = panel.copy()
        future.loc[0, "report_date"] = "2023-05-01"
        with self.assertRaisesRegex(ValueError, "cutoff"):
            PanelPriorSummary.from_frame(
                future,
                feature_cols=("y_pre__lag2",),
                target_col="y_pre",
                date_col="report_date",
                cutoff="2023-05-01",
            )

    def test_summary_json_and_parquet_round_trip(self) -> None:
        if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
            self.skipTest("Parquet engine is an optional local R&D dependency.")
        panel = generate_synthetic_panel(SyntheticConfig(n=20, seed=12), n_months=3)
        summary = PanelPriorSummary.from_frame(
            panel,
            feature_cols=("y_pre__lag2",),
            target_col="y_pre",
            date_col="report_date",
            cutoff="2023-05-01",
            entity_col="entity_id",
        )
        restored = PanelPriorSummary.from_dict(summary.to_dict())
        self.assertEqual(restored.to_dict(), summary.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.parquet"
            summary.to_parquet(path)
            self.assertEqual(PanelPriorSummary.from_parquet(path).to_dict(), summary.to_dict())

    def test_panel_prior_cupac_uses_pre_target_not_current_outcome_for_model_selection(self) -> None:
        historical = generate_synthetic_panel(SyntheticConfig(n=80, seed=21), n_months=5)
        summary = PanelPriorSummary.from_frame(
            historical,
            feature_cols=("y_pre__lag2", "x1__lag2", "x2__lag2"),
            target_col="y_pre",
            date_col="report_date",
            cutoff="2023-07-01",
            entity_col="entity_id",
        )
        data = generate_synthetic_ab(SyntheticConfig(n=160, seed=22))
        data["report_date"] = pd.Timestamp("2023-07-01")
        spec = ExperimentSpec(pre_target_col="y_pre", pre_treatment_cols=("y_pre", "x1", "x2"))
        transformer = PanelPriorCupacTransformer(summary=summary)
        first = transformer.fit_transform(data, spec)
        first_diagnostics = dict(transformer.diagnostics())
        changed = data.copy()
        changed["y"] += np.linspace(-100.0, 100.0, len(changed))
        second = transformer.fit_transform(changed, spec)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.isfinite(second).all())
        self.assertEqual(first_diagnostics["prior_strength"], transformer.diagnostics()["prior_strength"])
        self.assertEqual(first_diagnostics["validation_mse"], transformer.diagnostics()["validation_mse"])


if __name__ == "__main__":
    unittest.main()
