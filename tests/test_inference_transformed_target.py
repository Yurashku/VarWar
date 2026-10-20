"""Регрессия P0-1: non-native вывод обязан считать SE на преобразованном target."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import unittest

import numpy as np
import pandas as pd

from varwar.analysis import (
    DirectPointEvaluator,
    LegacyEstimatorPointAdapter,
    MaterializedPoint,
    RaceContext,
)
from varwar.analysis_registry import build_analysis_method_registry
from varwar.core import ExperimentSpec
from varwar.estimators import RegressionAdjustmentEstimator
from varwar.inference import (
    BootstrapTInference,
    DesignBasedNormalInference,
    DifferenceMeansPointEstimator,
    PooledStudentInference,
    StudentizedRandomizationInference,
    WelchInference,
)
from varwar.pipeline_execution import TransformingPointEvaluator
from varwar.target_registry import TARGET_REGISTRY


def _world(rng: np.random.Generator, n: int, effect: float = 0.0) -> pd.DataFrame:
    """Сильно прогностическая история: corr(y, y_pre) ≈ 0.9."""

    alpha = rng.normal(size=n)
    treatment = np.zeros(n)
    treatment[rng.permutation(n)[: n // 2]] = 1.0
    return pd.DataFrame(
        {
            "treatment": treatment,
            "y": alpha + 0.3 * rng.normal(size=n) + effect * treatment,
            "y_pre": alpha + 0.3 * rng.normal(size=n),
            "y_pre__lag2": alpha + 0.3 * rng.normal(size=n),
        }
    )


def _welch_se(data: pd.DataFrame, spec: ExperimentSpec) -> float:
    target = data[spec.target_col].to_numpy(float)
    treated = data[spec.treatment_col].to_numpy() == spec.treatment_value
    return math.sqrt(
        target[treated].var(ddof=1) / treated.sum()
        + target[~treated].var(ddof=1) / (~treated).sum()
    )


@dataclass
class _RecordingEvaluator:
    """Прозрачная обёртка, запоминающая каждую материализацию реплики."""

    inner: TransformingPointEvaluator | DirectPointEvaluator
    records: list[MaterializedPoint] = field(default_factory=list)

    def evaluate_point(self, data, spec, context):
        return self.evaluate_materialized(data, spec, context).point

    def materialize(self, data, spec, context):
        return self.inner.materialize(data, spec, context)

    def evaluate_materialized(self, data, spec, context):
        result = self.inner.evaluate_materialized(data, spec, context)
        self.records.append(result)
        return result


class TransformedTargetInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ExperimentSpec(pre_treatment_cols=("y_pre",))
        self.context = RaceContext()
        self.data = _world(np.random.default_rng(20261020), 200, effect=0.2)
        self.point = DifferenceMeansPointEstimator()
        self.cuped = TransformingPointEvaluator(
            self.point, (TARGET_REGISTRY["cuped_historical"].create,)
        )
        self.methods = build_analysis_method_registry(resamples=99)

    def test_materialize_returns_fitted_target_chain(self) -> None:
        frame, working_spec = self.cuped.materialize(self.data, self.spec, self.context)
        self.assertEqual(working_spec.target_col, "__varwar_target_stage_1")
        materialized = self.cuped.evaluate_materialized(
            self.data, self.spec, self.context
        )
        np.testing.assert_array_equal(
            frame[working_spec.target_col], materialized.data[working_spec.target_col]
        )
        direct = DirectPointEvaluator(self.point)
        identity_frame, identity_spec = direct.materialize(
            self.data, self.spec, self.context
        )
        self.assertIs(identity_frame, self.data)
        self.assertIs(identity_spec, self.spec)

    def test_cuped_welch_se_uses_adjusted_column(self) -> None:
        method = self.methods["diff_means__welch"]
        adjusted = method.analyze(
            self.data, self.spec, self.context, point_evaluator=self.cuped
        )
        theta = adjusted.diagnostics["target_stage_1_theta"]
        self.assertGreater(abs(theta), 0.5)
        frame, working_spec = self.cuped.materialize(
            self.context.analysis_frame(self.data, self.spec), self.spec, self.context
        )
        self.assertAlmostEqual(
            adjusted.standard_error, _welch_se(frame, working_spec), delta=1e-12
        )
        identity = method.analyze(self.data, self.spec, self.context)
        self.assertAlmostEqual(
            identity.standard_error, _welch_se(self.data, self.spec), delta=1e-12
        )
        self.assertLess(adjusted.standard_error, 0.6 * identity.standard_error)
        self.assertEqual(
            adjusted.diagnostics["inference_target_col"], "__varwar_target_stage_1"
        )

    def test_all_analytic_procedures_use_adjusted_column(self) -> None:
        materialized = self.cuped.evaluate_materialized(
            self.data, self.spec, self.context
        )
        welch_se = _welch_se(materialized.data, materialized.spec)
        for procedure in (WelchInference(), DesignBasedNormalInference()):
            # Без переданной материализации процедура строит её сама.
            result = procedure.infer(
                self.cuped, materialized.point, self.data, self.spec, self.context
            )
            self.assertAlmostEqual(result.standard_error, welch_se, delta=1e-12)
        pooled = PooledStudentInference().infer(
            self.cuped,
            materialized.point,
            self.data,
            self.spec,
            self.context,
            materialized=materialized,
        )
        raw_pooled = PooledStudentInference().infer(
            DirectPointEvaluator(self.point),
            materialized.point,
            self.data,
            self.spec,
            self.context,
        )
        self.assertLess(pooled.standard_error, 0.6 * raw_pooled.standard_error)

    def test_studentized_randomization_replicate_denominators_are_adjusted(self) -> None:
        recorder = _RecordingEvaluator(self.cuped)
        procedure = StudentizedRandomizationInference(n_resamples=99, random_state=3)
        distribution = procedure.replicate_distribution(
            recorder, self.data, self.spec, self.context
        )
        self.assertEqual(len(recorder.records), 99)
        expected = np.array(
            [_welch_se(record.data, record.spec) for record in recorder.records]
        )
        raw = np.array([_welch_se(record.data, self.spec) for record in recorder.records])
        np.testing.assert_allclose(distribution.standard_errors, expected, atol=1e-12)
        np.testing.assert_allclose(
            distribution.statistics, distribution.effects / expected, atol=1e-12
        )
        self.assertTrue(np.all(expected < 0.6 * raw))
        self.assertTrue(
            all(record.spec.target_col == "__varwar_target_stage_1" for record in recorder.records)
        )
        result = procedure.infer(recorder, recorder.records[0].point, self.data, self.spec, self.context)
        self.assertEqual(result.diagnostics["studentization"], "welch_on_materialized_target")

    def test_bootstrap_t_replicate_denominators_are_adjusted(self) -> None:
        recorder = _RecordingEvaluator(self.cuped)
        observed = self.cuped.evaluate_point(self.data, self.spec, self.context)
        procedure = BootstrapTInference(n_resamples=99, random_state=5)
        distribution = procedure.replicate_distribution(
            recorder, observed, self.data, self.spec, self.context
        )
        expected = np.array(
            [_welch_se(record.data, record.spec) for record in recorder.records]
        )
        np.testing.assert_allclose(distribution.standard_errors, expected, atol=1e-12)
        np.testing.assert_allclose(
            distribution.statistics,
            (distribution.effects - observed.effect) / expected,
            atol=1e-12,
        )
        result = procedure.infer(
            self.cuped, observed, self.data, self.spec, self.context
        )
        materialized = self.cuped.evaluate_materialized(self.data, self.spec, self.context)
        self.assertAlmostEqual(
            result.standard_error,
            _welch_se(materialized.data, materialized.spec),
            delta=1e-12,
        )

    def test_native_estimator_is_studentized_by_its_own_replicate_se(self) -> None:
        lin = LegacyEstimatorPointAdapter(RegressionAdjustmentEstimator())
        recorder = _RecordingEvaluator(DirectPointEvaluator(lin))
        procedure = StudentizedRandomizationInference(n_resamples=99, random_state=11)
        distribution = procedure.replicate_distribution(
            recorder, self.data, self.spec, self.context
        )
        native = np.array(
            [record.point.native_payload.standard_error for record in recorder.records]
        )
        np.testing.assert_allclose(distribution.standard_errors, native, atol=1e-12)
        observed = recorder.inner.evaluate_point(self.data, self.spec, self.context)
        result = procedure.infer(
            recorder.inner, observed, self.data, self.spec, self.context
        )
        self.assertEqual(result.standard_error, observed.native_payload.standard_error)
        self.assertEqual(result.diagnostics["studentization"], "native_standard_error")

    def test_native_inference_keeps_native_se_on_transformed_target(self) -> None:
        method = self.methods["lin_regression__hc2"]
        evaluator = TransformingPointEvaluator(
            method.point_estimator, (TARGET_REGISTRY["cuped_historical"].create,)
        )
        result = method.analyze(
            self.data, self.spec, self.context, point_evaluator=evaluator
        )
        frame, working_spec = evaluator.materialize(
            self.context.analysis_frame(self.data, self.spec), self.spec, self.context
        )
        reference = RegressionAdjustmentEstimator(covariance_type="HC2").estimate(
            frame, working_spec
        )
        self.assertAlmostEqual(result.standard_error, reference.standard_error, delta=1e-12)

    def test_cuped_welch_se_is_calibrated_monte_carlo(self) -> None:
        method = self.methods["diff_means__welch"]
        rng = np.random.default_rng(404)
        effects, errors, raw_errors = [], [], []
        for _ in range(200):
            data = _world(rng, 400)
            result = method.analyze(
                data, self.spec, self.context, point_evaluator=self.cuped
            )
            effects.append(result.effect)
            errors.append(result.standard_error)
            raw_errors.append(_welch_se(data, self.spec))
        empirical_sd = float(np.std(effects, ddof=1))
        ratio = float(np.mean(errors)) / empirical_sd
        self.assertGreaterEqual(ratio, 0.85)
        self.assertLessEqual(ratio, 1.15)
        # Сырой Welch SE (поведение до исправления) завышен более чем вдвое.
        self.assertGreater(float(np.mean(raw_errors)) / empirical_sd, 2.0)


if __name__ == "__main__":
    unittest.main()
