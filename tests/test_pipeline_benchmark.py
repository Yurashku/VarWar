from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from varwar.pipeline_benchmark import (
    SyntheticPipelineRaceConfig,
    format_pipeline_summary_ru,
    profile_synthetic_pipeline_memory,
    run_synthetic_pipeline_race,
    summarize_synthetic_pipeline_race,
)


def _summary_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        "welch_mean_diff": {
            "H0": ([0.00, 0.10, -0.10, 0.00], [0.50, 0.01, 0.50, 0.50]),
            "H1": ([0.05, 0.15, 0.25, 0.35], [0.30, 0.04, 0.01, 0.001]),
        },
        "cuped+hypex_ab": {
            "H0": ([0.00, 0.02, -0.02, 0.00], [0.50, 0.40, 0.40, 0.50]),
            "H1": ([0.17, 0.19, 0.21, 0.23], [0.01, 0.01, 0.01, 0.01]),
        },
    }
    for pipeline, hypotheses in values.items():
        for hypothesis, (effects, p_values) in hypotheses.items():
            truth = 0.0 if hypothesis == "H0" else 0.20
            for trial, (effect, p_value) in enumerate(zip(effects, p_values)):
                rows.append(
                    {
                        "data_scenario": "linear_gaussian",
                        "track": "full_traffic",
                        "pipeline": pipeline,
                        "hypothesis": hypothesis,
                        "true_effect": truth,
                        "trial": trial,
                        "status": "ok",
                        "effect": effect,
                        "standard_error": 0.05,
                        "p_value": p_value,
                        "ci_low": effect - 0.10,
                        "ci_high": effect + 0.10,
                        "analysis_rows": 1000,
                        "end_to_end_runtime_ms": 2.0,
                    }
                )
    return pd.DataFrame(rows)


class PipelineBenchmarkTests(unittest.TestCase):
    def test_summary_uses_h0_for_fpr_and_h1_for_efficiency(self) -> None:
        summary = summarize_synthetic_pipeline_race(_summary_fixture())
        indexed = summary.set_index("pipeline")
        self.assertAlmostEqual(
            indexed.loc["welch_mean_diff", "false_positive_rate"],
            0.25,
        )
        self.assertAlmostEqual(
            indexed.loc["cuped+hypex_ab", "power"],
            1.0,
        )
        self.assertGreater(
            indexed.loc[
                "cuped+hypex_ab", "variance_reduction_vs_baseline"
            ],
            0.9,
        )

    def test_russian_table_has_explicit_columns(self) -> None:
        russian = format_pipeline_summary_ru(
            summarize_synthetic_pipeline_race(_summary_fixture())
        )
        self.assertIn("Пайплайн", russian.columns)
        self.assertIn("FPR, % [95% ДИ]", russian.columns)
        self.assertIn("Мощность, % [95% ДИ]", russian.columns)
        self.assertIn("Снижение дисперсии, %", russian.columns)
        self.assertIn("Пик Python-памяти, МиБ", russian.columns)

    def test_small_race_uses_only_full_traffic(self) -> None:
        config = SyntheticPipelineRaceConfig(
            n_trials=2,
            n=400,
            n_strata=4,
            population_size=8,
            max_generations=4,
            elite_size=2,
            patience=2,
            min_macro_stratum_size=4,
            prefer_pypi_hypex=False,
        )
        raw = run_synthetic_pipeline_race(config)
        self.assertEqual(set(raw["track"]), {"full_traffic"})
        baseline = raw.loc[raw["pipeline"] == "welch_mean_diff"]
        self.assertTrue((baseline["analysis_rows"] == config.n).all())
        self.assertEqual(set(raw["hypothesis"]), {"H0", "H1"})
        self.assertNotIn("fixed_budget", set(raw["track"]))

    def test_memory_profile_is_separate_and_complete(self) -> None:
        config = SyntheticPipelineRaceConfig(
            n_trials=3,
            n=300,
            n_strata=4,
            population_size=8,
            max_generations=3,
            elite_size=2,
            patience=2,
            min_macro_stratum_size=4,
            prefer_pypi_hypex=False,
        )
        profile = profile_synthetic_pipeline_memory(config)
        self.assertEqual(set(profile["memory_profile_trials"]), {1})
        self.assertTrue((profile["peak_python_memory_bytes"] > 0).all())
        self.assertEqual(profile["pipeline"].nunique(), 15)


if __name__ == "__main__":
    unittest.main()
