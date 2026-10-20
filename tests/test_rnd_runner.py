from __future__ import annotations

import json
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from varwar.analysis import PipelineStatus
from varwar.analysis_registry import build_analysis_method_registry
from varwar.rnd_runner import (
    RNDRunConfig,
    aggregate_rnd_results,
    build_rnd_pipeline_matrix,
    run_rnd,
    select_rnd_leaders,
    trial_seeds,
)


_P = "adapt.identity__structure.none__assign.complete_randomization__target.{}__analysis.{}"
WELCH = _P.format("identity", "diff_means__welch")
CUPED = _P.format("cuped_historical", "diff_means__welch")
LIN = _P.format("identity", "lin_regression__hc2")
CUPAC_AB = _P.format("cupac_linear_A_plus_B", "diff_means__welch")
PANEL_A = _P.format("panel_prior_cupac_A", "diff_means__welch")
_NON_RESULT_COLUMNS = (
    "runtime_median_ms",
    "runtime_p95_ms",
    "peak_memory_mb",
    "peak_memory_mb_max",
    "executed_context",
)


def _synthetic_trials(n_trials: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for effect in (0.0, 0.1, 0.2):
        for trial in range(n_trials):
            estimate = effect + rng.normal(0.0, 0.1)
            rows.append(
                {
                    "pipeline": WELCH,
                    "world_id": "w",
                    "information_mode": "experiment_only",
                    "temporal_access": "pre_treatment_only",
                    "ranking_estimand_id": "sample_ate",
                    "assignment": "complete_randomization",
                    "structure": "none",
                    "target_transformer": "identity",
                    "analysis_method": "diff_means__welch",
                    "effect_size": effect,
                    "hypothesis": "H0" if effect == 0.0 else "H1",
                    "trial": trial,
                    "status": "ran",
                    "effect": estimate,
                    "standard_error": 0.1,
                    "p_value": float(2 * __import__("scipy").stats.norm.sf(abs(estimate / 0.1))),
                    "ci_low": estimate - 0.196,
                    "ci_high": estimate + 0.196,
                    "n_control": 100,
                    "n_treatment": 100,
                    "true_sample_ate": effect,
                    "true_population_ate": effect,
                    "true_native_estimand": np.nan,
                    "theory_grade": "A",
                    "causal_safety": "safe",
                    "runtime_ms": 1.0,
                }
            )
    return pd.DataFrame(rows)


class RNDRunnerTests(unittest.TestCase):
    def test_full_matrix_preserves_unavailable_and_temporal_contracts(self) -> None:
        registry = build_analysis_method_registry(
            population_weights={"s": 1.0},
            population_moments={"x1": 0.0},
            resamples=99,
            prefer_pypi_hypex=False,
        )
        matrix = build_rnd_pipeline_matrix(registry)
        statuses = {row.status for row in matrix}
        self.assertIn(PipelineStatus.ELIGIBLE, statuses)
        self.assertIn(PipelineStatus.UNAVAILABLE, statuses)
        self.assertIn(PipelineStatus.TEMPORAL_CONTRACT_VIOLATION, statuses)
        ids = {
            row.components.target_id
            for row in matrix
            if row.components.target_id.endswith("A_plus_B_plus_C")
        }
        self.assertGreaterEqual(len(ids), 5)
        routed = {
            row.components.analysis_method_id
            for row in matrix
            if row.status in {PipelineStatus.ELIGIBLE, PipelineStatus.POINT_ONLY}
        }
        self.assertEqual(set(registry), routed)

    def test_small_run_writes_numeric_ground_truth_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_rnd(
                RNDRunConfig(
                    n_trials=2,
                    resamples=99,
                    max_executable_pipelines=1,
                ),
                directory,
            )
            self.assertFalse(artifacts.trial_results.empty)
            self.assertIn("true_sample_ate", artifacts.trial_results)
            self.assertIn("h0_rejection_ci_high", artifacts.aggregates)
            self.assertIn("scientific_sample_size_ok", artifacts.aggregates)
            self.assertFalse(artifacts.aggregates["scientific_sample_size_ok"].any())
            self.assertTrue(
                artifacts.aggregates["inference_quality"].eq("warning").all()
            )
            self.assertTrue(artifacts.leaders.empty)
            self.assertTrue((artifacts.output_dir / "trial_results.parquet").exists())
            paired_seeds = artifacts.trial_results.groupby("trial")["seed"].nunique()
            self.assertTrue(paired_seeds.eq(1).all())
            self.assertEqual(
                artifacts.trial_results.groupby("trial")["seed"].first().nunique(),
                2,
            )

    def test_pipeline_uses_worst_target_and_analysis_safety(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_rnd(
                RNDRunConfig(
                    n_trials=2,
                    resamples=99,
                    pipeline_ids=(
                        "adapt.identity__structure.none__assign.complete_randomization"
                        "__target.cupac_linear_A_plus_B_plus_C"
                        "__analysis.diff_means__welch",
                    ),
                ),
                directory,
            )
            ran = artifacts.trial_results.loc[
                artifacts.trial_results["status"] == PipelineStatus.RAN.value
            ]
            self.assertFalse(ran.empty)
            self.assertTrue(ran["causal_safety"].eq("post_treatment_risk").all())
            self.assertTrue(ran["theory_grade"].eq("D").all())


    def test_trial_seeds_depend_only_on_world_and_trial(self) -> None:
        a = trial_seeds(20261001, "gaussian_r2_04_n200", 0)
        self.assertEqual(a, trial_seeds(20261001, "gaussian_r2_04_n200", 0))
        self.assertNotEqual(a["world"], trial_seeds(20261001, "contaminated", 0)["world"])
        self.assertNotEqual(a["world"], trial_seeds(20261001, "gaussian_r2_04_n200", 1)["world"])
        self.assertEqual(len(set(a.values())), 3)

    def test_seed_independent_of_world_order(self) -> None:
        config = RNDRunConfig(n_trials=2, pipeline_ids=(WELCH,), memory_profile_trials=0)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            both = run_rnd(
                RNDRunConfig(**{**config.__dict__, "world_ids": ("gaussian_r2_04_n200", "contaminated")}),
                first,
            ).trial_results
            alone = run_rnd(
                RNDRunConfig(**{**config.__dict__, "world_ids": ("contaminated",)}),
                second,
            ).trial_results
        keys = ["pipeline", "information_mode", "temporal_access", "effect_size", "trial"]
        subset = both.loc[both["world_id"] == "contaminated"].sort_values(keys).reset_index(drop=True)
        alone = alone.sort_values(keys).reset_index(drop=True)
        for column in ("seed", "assignment_seed", "effect", "standard_error", "true_sample_ate"):
            np.testing.assert_array_equal(subset[column].to_numpy(), alone[column].to_numpy())
        world_seeds = both.groupby("world_id")["seed"].first()
        self.assertEqual(world_seeds.nunique(), 2)

    def test_broadcast_equals_honest_execution(self) -> None:
        pipelines = (WELCH, CUPED, LIN, CUPAC_AB, PANEL_A)
        base = dict(n_trials=2, pipeline_ids=pipelines, memory_profile_trials=1)
        with tempfile.TemporaryDirectory() as fast, tempfile.TemporaryDirectory() as honest:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shared = run_rnd(RNDRunConfig(**base, broadcast_mode_invariant=True), fast)
                plain = run_rnd(RNDRunConfig(**base, broadcast_mode_invariant=False), honest)
            shared_meta = json.loads((shared.output_dir / "run_metadata.json").read_text())
            plain_meta = json.loads((plain.output_dir / "run_metadata.json").read_text())
        self.assertLess(shared_meta["n_pipeline_executions"], plain_meta["n_pipeline_executions"])
        self.assertIn("finalists", shared_meta)
        self.assertIn("selection_rule", shared_meta)
        self.assertEqual(len(shared.trial_results), len(plain.trial_results))
        invariant = shared.trial_results.loc[shared.trial_results["pipeline"] == WELCH]
        self.assertTrue(invariant["mode_invariant"].all())
        self.assertEqual(int((~invariant["broadcast"]).sum()), 2 * 2)  # effects x trials
        sensitive = shared.trial_results.loc[shared.trial_results["pipeline"] == CUPAC_AB]
        self.assertEqual(set(sensitive["context_sensitivity"]), {"temporal"})
        keys = ["pipeline", "information_mode", "temporal_access", "ranking_estimand_id", "effect_size"]
        left = shared.aggregates.drop(columns=list(_NON_RESULT_COLUMNS)).sort_values(keys).reset_index(drop=True)
        right = plain.aggregates.drop(columns=list(_NON_RESULT_COLUMNS)).sort_values(keys).reset_index(drop=True)
        self.assertEqual(list(left.columns), list(right.columns))
        self.assertGreater(len(left), 0)
        # Linear algebra may differ in the last ulp between calls (threaded
        # BLAS); everything else, including every verdict, must be identical.
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-9, atol=1e-12)
        self.assertTrue(shared.aggregates["mde_80"].notna().all())
        self.assertTrue(shared.aggregates["peak_memory_mb"].notna().all())

    def test_one_failing_effect_keeps_h0_and_other_effects(self) -> None:
        trials = _synthetic_trials()
        failing = trials["effect_size"] == 0.1
        trials.loc[failing, "status"] = "failed"
        trials.loc[failing, ["effect", "p_value", "ci_low", "ci_high"]] = np.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            aggregates = aggregate_rnd_results(trials)
        self.assertTrue(any("could not be summarised" in str(item.message) for item in caught))
        by_effect = aggregates.set_index("effect_size")
        self.assertEqual(set(by_effect.index), {0.1, 0.2})
        self.assertEqual(by_effect.loc[0.1, "inference_quality"], "bad")
        self.assertEqual(by_effect.loc[0.1, "verdict_reasons"], "execution_failure")
        self.assertAlmostEqual(float(by_effect.loc[0.1, "failure_rate"]), 0.5)
        self.assertEqual(float(by_effect.loc[0.2, "failure_rate"]), 0.0)
        self.assertTrue(np.isfinite(by_effect.loc[0.2, "h0_rejection_rate"]))
        self.assertTrue(np.isfinite(by_effect.loc[0.2, "mde_80"]))
        expected = (
            __import__("scipy").stats.t.ppf(0.975, 198) + __import__("scipy").stats.t.ppf(0.8, 198)
        ) * 0.1
        self.assertAlmostEqual(float(by_effect.loc[0.2, "mde_80"]), expected, places=10)

    def test_partial_failures_and_nan_pvalues_are_warnings(self) -> None:
        trials = _synthetic_trials(n_trials=120)
        h1 = trials.index[(trials["effect_size"] == 0.2)]
        trials.loc[h1[:3], "status"] = "failed"
        trials.loc[h1[3:5], "p_value"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aggregates = aggregate_rnd_results(trials)
        row = aggregates.loc[aggregates["effect_size"] == 0.2].iloc[0]
        self.assertGreater(float(row["failure_rate"]), 0.0)
        self.assertGreater(float(row["nan_pvalue_rate"]), 0.0)
        self.assertIn("execution_failures", row["verdict_reasons"])
        self.assertIn("nan_pvalues", row["verdict_reasons"])
        self.assertNotEqual(row["inference_quality"], "good")

    def test_leaders_rank_within_effect_size(self) -> None:
        frame = pd.DataFrame(
            {
                "pipeline": ["a", "b", "a", "b"],
                "world_id": "w",
                "information_mode": "experiment_only",
                "temporal_access": "pre_treatment_only",
                "ranking_estimand_id": "sample_ate",
                "effect_size": [0.1, 0.1, 0.2, 0.2],
                "inference_quality": "good",
                "scientific_sample_size_ok": True,
                "h1_rmse": [1.0, 2.0, 1.0, 2.0],
                "h1_rejection_rate": [0.3, 0.2, 0.9, 0.8],
                "runtime_median_ms": 1.0,
            }
        )
        leaders = select_rnd_leaders(frame)
        self.assertEqual(leaders.groupby("effect_size")["rank_power"].apply(sorted).tolist(), [[1.0, 2.0], [1.0, 2.0]])


    def test_spec_columns_follow_world_declaration(self) -> None:
        from varwar.rnd_runner import _spec
        from varwar.synthetic_v3 import WORLD_CATALOG

        legacy_cov = ("x1", "x2", "x3", "x4", "region", "device")
        legacy_pre = ("pre_target_lag1", "x1", "x2", "x3", "x4")
        for world_id, world_spec in WORLD_CATALOG.items():
            spec = _spec(None, world_spec)
            if getattr(world_spec, "noise_in_covariates", False):
                self.assertGreater(len(spec.covariate_cols), len(legacy_cov), world_id)
                self.assertTrue(any(c.startswith("noise_") for c in spec.pre_treatment_cols))
            else:
                self.assertEqual(spec.covariate_cols, legacy_cov, world_id)
                self.assertEqual(spec.pre_treatment_cols, legacy_pre, world_id)

    def test_estimand_tracks_use_exact_null_worlds(self) -> None:
        pate = _P.format("identity", "lin_regression__hc2_pate")
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_rnd(
                RNDRunConfig(
                    world_ids=("population_shift",),
                    n_trials=2,
                    pipeline_ids=(WELCH, pate),
                    memory_profile_trials=0,
                ),
                directory,
            )
        trials = artifacts.trial_results
        h0 = trials.loc[(trials["hypothesis"] == "H0") & (trials["status"] == "ran")]
        sate = h0.loc[h0["ranking_estimand_id"] == "sample_ate"]
        pop = h0.loc[h0["ranking_estimand_id"] == "population_ate"]
        self.assertFalse(sate.empty)
        self.assertFalse(pop.empty)
        self.assertLess(float(sate["true_sample_ate"].abs().max()), 1e-9)
        self.assertLess(float(pop["true_population_ate"].abs().max()), 1e-9)
        self.assertEqual(set(sate["effect_modifier_centering"]), {"cohort"})
        self.assertEqual(set(pop["effect_modifier_centering"]), {"population"})
        # Both tracks share world seed, cohort and assignment.
        per_trial = h0.groupby("trial")[["seed", "assignment_seed"]].nunique()
        self.assertTrue((per_trial == 1).all().all())
        self.assertTrue(artifacts.aggregates["h0_is_null"].all())

    def test_inference_failure_diagnostic_counts_as_failure(self) -> None:
        trials = _synthetic_trials(n_trials=120)
        trials["diag_inference_failure"] = False
        h1 = trials.index[trials["effect_size"] == 0.2]
        trials.loc[h1[:4], "diag_inference_failure"] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aggregates = aggregate_rnd_results(trials)
        row = aggregates.loc[aggregates["effect_size"] == 0.2].iloc[0]
        other = aggregates.loc[aggregates["effect_size"] == 0.1].iloc[0]
        self.assertAlmostEqual(float(row["inference_failure_rate"]), 4 / 240)
        self.assertAlmostEqual(float(row["failure_rate"]), 4 / 240)
        self.assertIn("inference_failures", row["verdict_reasons"])
        self.assertNotEqual(row["inference_quality"], "good")
        self.assertFalse(bool(row["validity_confirmed"]))
        self.assertEqual(float(other["failure_rate"]), 0.0)

    def test_fast_only_skips_gbm_and_screening_list_is_executable(self) -> None:
        from pathlib import Path

        from varwar.rnd_runner import _cached_pipeline_matrix, _execution_rows

        registry = build_analysis_method_registry(
            population_weights={"s": 1.0},
            population_moments={"x1": 0.0},
            resamples=99,
        )
        matrix, _ = _cached_pipeline_matrix(registry)
        fast = _execution_rows(matrix, RNDRunConfig(fast_only=True))
        self.assertFalse(any("gbm" in row.components.analysis_method_id for row in fast))
        full = _execution_rows(matrix, RNDRunConfig(stage="screening", include_hypex_assignment=True))
        executable = {row.components.pipeline_id for row in full}
        self.assertIn(_P.format("identity", "ml_rate__crossfit_gbm_hc2"), executable)
        listed = [
            line.split("#", 1)[0].strip()
            for line in (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "rnd_v3_1_screening_pipelines.txt"
            ).read_text(encoding="utf-8").splitlines()
        ]
        listed = [line for line in listed if line]
        self.assertGreaterEqual(len(listed), 10)
        self.assertEqual([pid for pid in listed if pid not in executable], [])


if __name__ == "__main__":
    unittest.main()


class FinalistExclusionRegressionTests(unittest.TestCase):
    """Пайплайн с полным отказом в одном мире не может стать финалистом."""

    def test_total_failure_row_excludes_pipeline(self):
        import pandas as pd
        from varwar.rnd_runner import select_confirmation_finalists

        base = {
            "information_mode": "experiment_only", "temporal_access": "pre_treatment_only",
            "ranking_estimand_id": "sample_ate", "effect_size": 0.2, "causal_safety": "safe",
            "target_transformer": "identity", "assignment": "complete_randomization",
            "structure": "none", "runtime_median_ms": 1.0,
        }
        rows = []
        for world in ("w1", "w2"):
            rows.append({**base, "world_id": world, "pipeline": "welch", "analysis_method": "diff_means__welch",
                         "h1_rmse": 1.0, "rmse_ratio_vs_welch_cr": 1.0, "inference_quality": "good",
                         "failure_rate": 0.0, "scientific_sample_size_ok": True})
            rows.append({**base, "world_id": world, "pipeline": "good_lin", "analysis_method": "lin",
                         "h1_rmse": 0.8, "rmse_ratio_vs_welch_cr": 0.8, "inference_quality": "good",
                         "failure_rate": 0.0, "scientific_sample_size_ok": True})
        rows.append({**base, "world_id": "w1", "pipeline": "flaky", "analysis_method": "flaky",
                     "h1_rmse": 0.5, "rmse_ratio_vs_welch_cr": 0.5, "inference_quality": "good",
                     "failure_rate": 0.0, "scientific_sample_size_ok": True})
        rows.append({**base, "world_id": "w2", "pipeline": "flaky", "analysis_method": "flaky",
                     "h1_rmse": float("nan"), "rmse_ratio_vs_welch_cr": float("nan"),
                     "inference_quality": "bad", "failure_rate": 1.0,
                     "scientific_sample_size_ok": False})
        finalists = select_confirmation_finalists(pd.DataFrame(rows))
        picked = set(finalists.loc[finalists["role"] != "anchor", "pipeline"])
        self.assertNotIn("flaky", picked)
        self.assertIn("good_lin", picked)
        self.assertEqual(int(finalists.loc[finalists["pipeline"] == "good_lin", "n_worlds_in_track"].iloc[0]), 2)
