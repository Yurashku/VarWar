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
    pipeline_definitions,
    profile_synthetic_pipeline_memory,
    run_synthetic_pipeline_race,
    summarize_synthetic_pipeline_race,
)
from varwar.method_catalog import method_catalog


def _summary_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        "welch": {
            "H0": ([0.00, 0.10, -0.10, 0.00], [0.50, 0.01, 0.50, 0.50]),
            "H1": ([0.05, 0.15, 0.25, 0.35], [0.30, 0.04, 0.01, 0.001]),
        },
        "cuped_control__hypex_ab": {
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
                        "pipeline_name_ru": pipeline,
                        "components": pipeline,
                        "estimand": "ATE",
                        "pipeline_tier": "confirmatory",
                        "data_contract": "fixture",
                        "pipeline_notes": "",
                        "predecessor_pipeline": (
                            "welch"
                            if pipeline == "cuped_control__hypex_ab"
                            else None
                        ),
                        "scenario_name_ru": "Линейный",
                        "scenario_purpose_ru": "fixture",
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
    def test_every_pipeline_id_is_in_method_catalog(self) -> None:
        config = SyntheticPipelineRaceConfig()
        known_ids = {record.method_id for record in method_catalog()}
        pipeline_ids = {
            definition.pipeline_id for definition in pipeline_definitions(config)
        }
        self.assertTrue(pipeline_ids.issubset(known_ids))

    def test_summary_uses_h0_for_fpr_and_h1_for_efficiency(self) -> None:
        summary = summarize_synthetic_pipeline_race(_summary_fixture())
        indexed = summary.set_index("pipeline")
        self.assertAlmostEqual(
            indexed.loc["welch", "false_positive_rate"],
            0.25,
        )
        self.assertAlmostEqual(
            indexed.loc["cuped_control__hypex_ab", "power"],
            1.0,
        )
        self.assertGreater(
            indexed.loc[
                "cuped_control__hypex_ab", "variance_reduction_vs_welch"
            ],
            0.9,
        )

    def test_diagnostic_tolerance_is_not_replaced_by_non_rejection(self) -> None:
        trials = _summary_fixture()
        shifted = (
            (trials["pipeline"] == "welch")
            & (trials["hypothesis"] == "H1")
        )
        trials.loc[shifted, ["effect", "ci_low", "ci_high"]] += 0.03

        row = (
            summarize_synthetic_pipeline_race(trials)
            .set_index("pipeline")
            .loc["welch"]
        )

        self.assertTrue(row["bias_calibrated"])
        self.assertFalse(row["bias_within_tolerance"])
        self.assertFalse(row["passes_diagnostic_screen"])
        self.assertEqual(row["diagnostic_status"], "смещение выше допуска")

    def test_changed_metric_hides_incomparable_bias_and_coverage(self) -> None:
        trials = _summary_fixture()
        changed = trials["pipeline"] == "welch"
        trials.loc[changed, "pipeline_tier"] = "changed_metric"
        trials.loc[changed, "estimand"] = "ATE винзоризированного target"

        row = (
            summarize_synthetic_pipeline_race(trials)
            .set_index("pipeline")
            .loc["welch"]
        )

        self.assertTrue(pd.isna(row["true_effect"]))
        self.assertTrue(pd.isna(row["bias_h1"]))
        self.assertTrue(pd.isna(row["coverage_h1"]))
        self.assertFalse(row["passes_diagnostic_screen"])
        self.assertEqual(row["diagnostic_status"], "другой estimand")

    def test_russian_table_has_explicit_columns(self) -> None:
        russian = format_pipeline_summary_ru(
            summarize_synthetic_pipeline_race(_summary_fixture())
        )
        self.assertIn("Пайплайн", russian.columns)
        self.assertIn("FPR, % [95% ДИ]", russian.columns)
        self.assertIn("Мощность, % [95% ДИ]", russian.columns)
        self.assertIn("Снижение дисперсии к Welch, %", russian.columns)
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
            scenario_ids=("linear_gaussian",),
        )
        raw = run_synthetic_pipeline_race(config)
        baseline = raw.loc[raw["pipeline"] == "welch"]
        self.assertTrue((baseline["analysis_rows"] == config.n).all())
        self.assertEqual(set(raw["hypothesis"]), {"H0", "H1"})
        self.assertEqual(set(raw["data_scenario"]), {"linear_gaussian"})
        self.assertNotIn("neyman", set(raw["preprocessor"]))

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
            scenario_ids=("linear_gaussian",),
        )
        profile = profile_synthetic_pipeline_memory(config)
        self.assertEqual(set(profile["memory_profile_trials"]), {1})
        self.assertTrue((profile["peak_python_memory_bytes"] > 0).all())
        self.assertEqual(profile["pipeline"].nunique(), 17)

    def test_pipeline_filter_runs_only_requested_rows(self) -> None:
        config = SyntheticPipelineRaceConfig(
            n_trials=1,
            n=300,
            scenario_ids=("linear_gaussian",),
            include_stratified=True,
            prefer_pypi_hypex=False,
            pipeline_ids=("welch", "ga_strata__stratified"),
            population_size=8,
            max_generations=3,
            elite_size=2,
            patience=2,
            min_macro_stratum_size=4,
        )
        raw = run_synthetic_pipeline_race(config)
        self.assertEqual(
            set(raw["pipeline"]),
            {"welch", "ga_strata__stratified"},
        )


if __name__ == "__main__":
    unittest.main()
