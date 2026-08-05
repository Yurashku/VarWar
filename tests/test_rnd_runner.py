from __future__ import annotations

import tempfile
import unittest

from varwar.analysis import PipelineStatus
from varwar.analysis_registry import build_analysis_method_registry
from varwar.rnd_runner import (
    RNDRunConfig,
    build_rnd_pipeline_matrix,
    run_rnd,
)


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


if __name__ == "__main__":
    unittest.main()
