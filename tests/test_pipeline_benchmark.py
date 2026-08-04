from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from varwar.core import ExperimentSpec

from varwar.pipeline_benchmark import (
    SyntheticPipelineRaceConfig,
    format_pipeline_summary_ru,
    pipeline_definitions,
    profile_synthetic_pipeline_memory,
    run_synthetic_pipeline_race,
    select_best_pipelines,
    summarize_synthetic_pipeline_race,
)
from varwar.method_catalog import method_catalog
from varwar.pipelines import (
    build_cartesian_rct_pipelines,
    build_full_traffic_pipelines,
    build_stratified_pipelines,
)


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

    def test_every_implemented_estimator_has_a_pipeline_route(self) -> None:
        spec = ExperimentSpec(
            pre_treatment_cols=("x1",),
            numerator_col="numerator",
            denominator_col="denominator",
            instrument_col="assigned",
            received_treatment_col="received",
            stratum_col="stratum",
        )
        definitions = [
            candidate.definition
            for candidate in build_full_traffic_pipelines(
                spec,
                prefer_pypi_hypex=False,
            )
        ]
        definitions.extend(
            candidate.definition
            for candidate in build_stratified_pipelines(spec, {"example": 1.0})
        )
        covered = {
            component
            for definition in definitions
            for component in definition.components
        }
        covered.update(definition.pipeline_id for definition in definitions)
        implemented = {
            record.method_id
            for record in method_catalog()
            if record.category.value == "Estimator" and record.is_implemented
        }
        self.assertTrue(implemented.issubset(covered))

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
        self.assertEqual(
            profile["pipeline"].nunique(),
            len(pipeline_definitions(config)),
        )

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

    def test_trial_indices_keep_global_trial_labels(self) -> None:
        config = SyntheticPipelineRaceConfig(
            n_trials=3,
            n=300,
            scenario_ids=("linear_gaussian",),
            include_stratified=False,
            prefer_pypi_hypex=False,
            pipeline_ids=("welch",),
        )
        selected = run_synthetic_pipeline_race(config, trial_indices=(2,))
        self.assertEqual(set(selected["trial"]), {2})
        repeated = run_synthetic_pipeline_race(config, trial_indices=(2,))
        deterministic_columns = [
            column
            for column in selected.columns
            if "runtime" not in column and "memory" not in column
        ]
        pd.testing.assert_frame_equal(
            selected[deterministic_columns].reset_index(drop=True),
            repeated[deterministic_columns].reset_index(drop=True),
            check_dtype=False,
        )

    def test_population_information_race_has_sample_and_population_estimands(self) -> None:
        config = SyntheticPipelineRaceConfig(
            n_trials=1,
            n=800,
            scenario_ids=("population_shifted_strata",),
            include_stratified=False,
            prefer_pypi_hypex=False,
            pipeline_ids=(
                "stratified_mean_diff",
                "population_weighted_stratified",
                "calibration_reweighting",
            ),
        )
        raw = run_synthetic_pipeline_race(config)
        self.assertEqual(
            set(raw["pipeline"]),
            {
                "stratified_mean_diff",
                "population_weighted_stratified",
                "calibration_reweighting",
            },
        )
        self.assertTrue((raw["status"] == "ok").all())
        self.assertEqual(set(raw["true_effect"]), {0.0, config.alternative_effect})

    def test_race_keeps_information_and_temporal_modes_separate(self) -> None:
        config = SyntheticPipelineRaceConfig(
            n_trials=1,
            n=300,
            scenario_ids=("linear_gaussian",),
            include_stratified=False,
            prefer_pypi_hypex=False,
            pipeline_ids=(
                "welch",
                "cupac_linear_A__hypex_ab",
                "cupac_linear_A_plus_B__hypex_ab",
                "panel_prior_cupac_A__hypex_ab",
            ),
        )
        raw = run_synthetic_pipeline_race(config)
        self.assertEqual(set(raw["race_information_mode"]), {"experiment_only", "panel_prior"})
        modes = raw[["pipeline", "race_information_mode", "information_mode", "temporal_mode"]].drop_duplicates()
        self.assertEqual(
            set(modes.loc[modes["pipeline"] == "welch", "temporal_mode"]),
            {"pre_treatment_only"},
        )
        self.assertEqual(
            set(modes.loc[modes["pipeline"] == "cupac_linear_A_plus_B__hypex_ab", "temporal_mode"]),
            {"post_treatment_allowed"},
        )
        self.assertEqual(
            set(modes.loc[modes["pipeline"] == "panel_prior_cupac_A__hypex_ab", "race_information_mode"]),
            {"panel_prior"},
        )

    def test_cartesian_rct_product_has_explicit_component_contracts(self) -> None:
        spec = ExperimentSpec(
            unit_col="unit_id",
            pre_target_col="y_pre",
            covariate_cols=("y_pre", "x1", "x2"),
            pre_treatment_cols=("y_pre", "x1", "x2"),
            expert_safe_in_time_cols=("calendar_index",),
        )
        local = build_cartesian_rct_pipelines(
            spec, prefer_pypi_hypex=False, include_panel_prior=False
        )
        self.assertEqual(len(local), 48)
        self.assertEqual(len({candidate.definition.pipeline_id for candidate in local}), 48)
        self.assertTrue(
            any(candidate.definition.pipeline_id == "cartesian__welch__cuped_control" for candidate in local)
        )
        self.assertTrue(
            all(candidate.definition.information_mode == "experiment_only" for candidate in local)
        )
        panel = build_cartesian_rct_pipelines(
            spec, prefer_pypi_hypex=False, include_panel_prior=True
        )
        self.assertEqual(len(panel), 72)
        self.assertTrue(
            all(
                candidate.definition.information_mode == "panel_prior"
                for candidate in panel
                if "panel_prior" in candidate.definition.pipeline_id
            )
        )
        self.assertTrue(
            any(
                candidate.definition.temporal_mode == "post_treatment_allowed"
                for candidate in panel
            )
        )

    def test_best_pipeline_selection_never_crosses_information_or_time_contracts(self) -> None:
        rows = []
        for pipeline, info, temporal, sd, eligible in (
            ("a", "experiment_only", "pre_treatment_only", 0.10, True),
            ("b", "experiment_only", "pre_treatment_only", 0.08, True),
            ("c", "panel_prior", "pre_treatment_only", 0.01, True),
            ("d", "experiment_only", "post_treatment_allowed", 0.01, True),
        ):
            rows.append(
                {
                    "data_scenario": "linear_gaussian",
                    "race_information_mode": info,
                    "temporal_mode": temporal,
                    "estimand_id": "sample_ate",
                    "pipeline": pipeline,
                    "pipeline_name_ru": pipeline,
                    "pipeline_tier": "confirmatory",
                    "ranking_eligible": eligible,
                    "scientific_sample_size_ok": True,
                    "passes_diagnostic_screen": True,
                    "empirical_sd_h1": sd,
                    "variance_reduction_vs_welch": 0.1,
                    "median_runtime_ms": 1.0,
                }
            )
        selected = select_best_pipelines(pd.DataFrame(rows))
        winners = selected.set_index(
            ["race_information_mode", "temporal_mode"]
        )["selected_pipeline"]
        self.assertEqual(winners.loc["experiment_only", "pre_treatment_only"], "b")
        self.assertEqual(winners.loc["panel_prior", "pre_treatment_only"], "c")
        self.assertEqual(winners.loc["experiment_only", "post_treatment_allowed"], "d")

    def test_best_pipeline_selection_reports_insufficient_trials(self) -> None:
        row = pd.DataFrame(
            [
                {
                    "data_scenario": "linear_gaussian",
                    "race_information_mode": "experiment_only",
                    "temporal_mode": "pre_treatment_only",
                    "estimand_id": "sample_ate",
                    "pipeline": "welch",
                    "pipeline_name_ru": "Welch",
                    "pipeline_tier": "confirmatory",
                    "ranking_eligible": False,
                    "scientific_sample_size_ok": False,
                    "passes_diagnostic_screen": True,
                    "empirical_sd_h1": 0.1,
                    "variance_reduction_vs_welch": 0.0,
                    "median_runtime_ms": 1.0,
                }
            ]
        )
        selected = select_best_pipelines(row)
        self.assertEqual(
            selected.loc[0, "selection_status"],
            "недостаточно повторений для рейтинга",
        )


if __name__ == "__main__":
    unittest.main()
