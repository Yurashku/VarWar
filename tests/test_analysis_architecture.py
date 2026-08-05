from __future__ import annotations

import unittest

import pandas as pd

from varwar.analysis import (
    AnalysisMethod,
    CausalSafety,
    CompatibilityContract,
    InformationMode,
    LegacyEstimatorPointAdapter,
    NativeEstimatorInference,
    PipelineStatus,
    RaceContext,
    TemporalAccess,
    TheoryGrade,
)
from varwar.core import ExperimentSpec
from varwar.estimators import DifferenceInMeansEstimator
from varwar.pipeline_matrix import build_pipeline_matrix
from varwar.pipeline_execution import execute_pipeline_row


class AnalysisArchitectureTests(unittest.TestCase):
    def _method(self, estimand: str = "sample_ate") -> AnalysisMethod:
        estimator = DifferenceInMeansEstimator(name="welch")
        return AnalysisMethod(
            method_id="diff_means__welch",
            point_estimator=LegacyEstimatorPointAdapter(
                estimator,
                estimand_id=estimand,
            ),
            inference=NativeEstimatorInference(estimator),
            contract=CompatibilityContract(
                estimand_id=estimand,
                ranking_track=estimand,
                theory_grade=TheoryGrade.A,
                causal_safety=CausalSafety.SAFE,
                inference_requirements=frozenset({"native"}),
                primary_ranking_eligible=True,
            ),
        )

    def test_context_physically_hides_future_columns(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0, 1],
                "y": [0.0, 1.0],
                "x_pre": [1.0, 2.0],
                "x_b": [2.0, 3.0],
                "x_c": [3.0, 4.0],
                "post_proxy": [0.0, 1.0],
            }
        )
        spec = ExperimentSpec(
            pre_treatment_cols=("x_pre",),
            expert_safe_in_time_cols=("x_b",),
            gated_in_time_cols=("x_c",),
            unsafe_demo_cols=("post_proxy",),
            pre_target_col=None,
        )
        pre = RaceContext().analysis_frame(data, spec)
        self.assertEqual(set(pre), {"treatment", "y", "x_pre"})
        full = RaceContext(
            temporal_access=TemporalAccess.FULL_TIMELINE
        ).analysis_frame(data, spec)
        self.assertEqual(set(full), set(data))
        assignment = RaceContext().assignment_frame(data, spec)
        self.assertEqual(set(assignment), {"x_pre"})

    def test_analysis_method_runs_registered_pair(self) -> None:
        data = pd.DataFrame(
            {"treatment": [0, 0, 1, 1], "y": [0.0, 1.0, 2.0, 3.0]}
        )
        result = self._method().analyze(data, ExperimentSpec(), RaceContext())
        self.assertAlmostEqual(result.effect, 2.0)
        self.assertEqual(result.theory_grade, TheoryGrade.A)

    def test_matrix_keeps_estimand_mismatch_and_unavailable_rows(self) -> None:
        sample = self._method("sample_ate")
        panel_contract = CompatibilityContract(
            estimand_id="population_ate",
            ranking_track="population_ate",
            theory_grade=TheoryGrade.B,
            causal_safety=CausalSafety.CONDITIONAL,
            information_modes=frozenset({InformationMode.PANEL_PRIOR}),
            inference_requirements=frozenset({"native"}),
        )
        estimator = DifferenceInMeansEstimator(name="panel")
        panel = AnalysisMethod(
            "panel",
            LegacyEstimatorPointAdapter(estimator, "population_ate"),
            NativeEstimatorInference(estimator),
            panel_contract,
        )
        contexts = (
            RaceContext(ranking_estimand_id="sample_ate"),
            RaceContext(ranking_estimand_id="population_ate"),
        )
        rows = build_pipeline_matrix(
            adapter_ids=("identity",),
            structure_ids=("none",),
            assignment_ids=("complete_randomization",),
            target_ids=("identity",),
            analysis_methods={"sample": sample, "panel": panel},
            contexts=contexts,
        )
        self.assertEqual(len(rows), 4)
        self.assertIn(PipelineStatus.ESTIMAND_MISMATCH, {row.status for row in rows})
        self.assertIn(PipelineStatus.UNAVAILABLE, {row.status for row in rows})

    def test_point_only_keeps_effect_and_hides_untrusted_inference(self) -> None:
        method = self._method()
        method = AnalysisMethod(
            method.method_id,
            method.point_estimator,
            method.inference,
            CompatibilityContract(
                estimand_id="sample_ate",
                ranking_track="sample_ate",
                theory_grade=TheoryGrade.A,
                causal_safety=CausalSafety.SAFE,
                assignment_ids=frozenset({"complete_randomization"}),
                inference_requirements=frozenset({"native"}),
            ),
        )
        context = RaceContext(assignment_id="rerandomization_max_smd")
        rows = build_pipeline_matrix(
            adapter_ids=("identity",),
            structure_ids=("none",),
            assignment_ids=("rerandomization_max_smd",),
            target_ids=("identity",),
            analysis_methods={method.method_id: method},
            contexts=(context,),
        )
        self.assertEqual(rows[0].status, PipelineStatus.POINT_ONLY)
        data = pd.DataFrame(
            {"treatment": [0, 0, 1, 1], "y": [0.0, 1.0, 2.0, 3.0]}
        )
        result = execute_pipeline_row(rows[0], method, data, ExperimentSpec())
        self.assertEqual(result["status"], "point_only")
        self.assertAlmostEqual(result["effect"], 2.0)
        self.assertTrue(pd.isna(result["p_value"]))


if __name__ == "__main__":
    unittest.main()
