from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import stats

from varwar.analysis import (
    InferenceResult,
    MaterializedPoint,
    PointEstimate,
    PointEvaluator,
    RaceContext,
    evaluate_materialized,
)
from varwar.core import EstimateResult, ExperimentSpec, require_columns


def _groups(
    data: pd.DataFrame,
    spec: ExperimentSpec,
) -> tuple[np.ndarray, np.ndarray]:
    require_columns(data, [spec.treatment_col, spec.target_col])
    target = pd.to_numeric(data[spec.target_col], errors="coerce").to_numpy(float)
    treatment = np.asarray(data[spec.treatment_col])
    control = target[np.isfinite(target) & (treatment == spec.control_value)]
    treated = target[np.isfinite(target) & (treatment == spec.treatment_value)]
    if control.size < 2 or treated.size < 2:
        raise ValueError("Both groups need at least two finite observations.")
    return control, treated


def _welch_parameters(
    control: np.ndarray,
    treated: np.ndarray,
) -> tuple[float, float]:
    vc = float(control.var(ddof=1)) / control.size
    vt = float(treated.var(ddof=1)) / treated.size
    se = math.sqrt(vc + vt)
    denominator = vc * vc / (control.size - 1) + vt * vt / (treated.size - 1)
    df = (vc + vt) ** 2 / denominator if denominator > 0.0 else math.inf
    return se, df


def _observed_materialization(
    evaluator: PointEvaluator,
    observed: PointEstimate,
    data: pd.DataFrame,
    spec: ExperimentSpec,
    context: RaceContext,
    materialized: MaterializedPoint | None,
) -> MaterializedPoint:
    """Frame/spec наблюдённой точки: преобразованный target, а не сырой исход."""

    if materialized is not None:
        return materialized
    materialize = getattr(evaluator, "materialize", None)
    if not callable(materialize):
        raise TypeError(
            f"{type(evaluator).__name__} must implement materialize(data, spec, "
            "context) so that inference uses the transformed target."
        )
    frame, working_spec = materialize(data, spec, context)
    return MaterializedPoint(observed, frame, working_spec)


def _studentization_scale(
    materialized: MaterializedPoint,
) -> tuple[float, float, str]:
    """SE той же процедуры, что дала точку: native SE или Welch по рабочему target."""

    native = materialized.point.native_payload
    if isinstance(native, EstimateResult):
        df = native.diagnostics.get("degrees_of_freedom", math.nan)
        return float(native.standard_error), float(df), "native_standard_error"
    control, treated = _groups(materialized.data, materialized.spec)
    se, df = _welch_parameters(control, treated)
    return se, df, "welch_on_materialized_target"


@dataclass(frozen=True)
class ResamplingDistribution:
    """Реплики full-refit процедуры: точка, её собственный SE и pivot."""

    effects: np.ndarray
    standard_errors: np.ndarray
    statistics: np.ndarray
    degenerate_replicates: int


@dataclass(frozen=True)
class DifferenceMeansPointEstimator:
    method_id: str = "difference_in_means"
    estimand_id: str = "sample_ate"

    def estimate_point(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> PointEstimate:
        del context
        control, treated = _groups(data, spec)
        return PointEstimate(
            effect=float(treated.mean() - control.mean()),
            estimand_id=self.estimand_id,
            diagnostics={
                "point_formula": "mean_treatment_minus_mean_control",
                "n_control": int(control.size),
                "n_treatment": int(treated.size),
            },
        )


@dataclass(frozen=True)
class WelchInference:
    inference_id: str = "welch"
    capabilities: frozenset[str] = frozenset({"native", "unequal_variance"})

    def infer(
        self,
        evaluator: PointEvaluator,
        observed: PointEstimate,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
        *,
        materialized: MaterializedPoint | None = None,
    ) -> InferenceResult:
        working = _observed_materialization(
            evaluator, observed, data, spec, context, materialized
        )
        control, treated = _groups(working.data, working.spec)
        se, df = _welch_parameters(control, treated)
        statistic = observed.effect / se if se > 0.0 else math.nan
        critical = float(stats.t.ppf(1.0 - spec.alpha / 2.0, df))
        return InferenceResult(
            standard_error=se,
            statistic=statistic,
            p_value=float(2.0 * stats.t.sf(abs(statistic), df)),
            ci_low=float(observed.effect - critical * se),
            ci_high=float(observed.effect + critical * se),
            diagnostics={
                "degrees_of_freedom": df,
                "inference_family": "welch",
                "inference_target_col": working.spec.target_col,
            },
        )


@dataclass(frozen=True)
class PooledStudentInference:
    inference_id: str = "student_pooled"
    capabilities: frozenset[str] = frozenset({"native", "equal_variance"})

    def infer(
        self,
        evaluator: PointEvaluator,
        observed: PointEstimate,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
        *,
        materialized: MaterializedPoint | None = None,
    ) -> InferenceResult:
        working = _observed_materialization(
            evaluator, observed, data, spec, context, materialized
        )
        control, treated = _groups(working.data, working.spec)
        df = float(control.size + treated.size - 2)
        pooled = (
            (control.size - 1) * control.var(ddof=1)
            + (treated.size - 1) * treated.var(ddof=1)
        ) / df
        se = math.sqrt(float(pooled) * (1.0 / control.size + 1.0 / treated.size))
        statistic = observed.effect / se if se > 0.0 else math.nan
        critical = float(stats.t.ppf(1.0 - spec.alpha / 2.0, df))
        return InferenceResult(
            standard_error=se,
            statistic=statistic,
            p_value=float(2.0 * stats.t.sf(abs(statistic), df)),
            ci_low=float(observed.effect - critical * se),
            ci_high=float(observed.effect + critical * se),
            diagnostics={
                "degrees_of_freedom": df,
                "inference_family": "pooled_student",
                "inference_target_col": working.spec.target_col,
            },
        )


@dataclass(frozen=True)
class DesignBasedNormalInference:
    """Neyman-style variance with normal calibration, not Neyman allocation."""

    inference_id: str = "design_based_normal"
    capabilities: frozenset[str] = frozenset({"native", "design_based"})

    def infer(
        self,
        evaluator: PointEvaluator,
        observed: PointEstimate,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
        *,
        materialized: MaterializedPoint | None = None,
    ) -> InferenceResult:
        working = _observed_materialization(
            evaluator, observed, data, spec, context, materialized
        )
        control, treated = _groups(working.data, working.spec)
        se, _ = _welch_parameters(control, treated)
        statistic = observed.effect / se if se > 0.0 else math.nan
        critical = float(stats.norm.ppf(1.0 - spec.alpha / 2.0))
        return InferenceResult(
            standard_error=se,
            statistic=statistic,
            p_value=float(2.0 * stats.norm.sf(abs(statistic))),
            ci_low=float(observed.effect - critical * se),
            ci_high=float(observed.effect + critical * se),
            diagnostics={
                "inference_family": "design_based_normal",
                "inference_target_col": working.spec.target_col,
            },
        )


@dataclass(frozen=True)
class StudentizedRandomizationInference:
    """Перестановочный тест по pivot ``effect / SE`` той же full-refit процедуры.

    В каждой перестановке target-цепочка подгоняется заново, а знаменатель —
    SE именно этой реплики (native SE или Welch по её преобразованному target).
    """

    n_resamples: int = 999
    random_state: int = 42
    inference_id: str = "welch_studentized_randomization"
    capabilities: frozenset[str] = frozenset({"design_aware", "full_refit"})

    def replicate_distribution(
        self,
        evaluator: PointEvaluator,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> ResamplingDistribution:
        if self.n_resamples < 99:
            raise ValueError("n_resamples must be at least 99.")
        treatment = np.asarray(data[spec.treatment_col]).copy()
        rng = np.random.default_rng(self.random_state)
        effects = np.empty(self.n_resamples, dtype=float)
        scales = np.empty(self.n_resamples, dtype=float)
        statistics = np.empty(self.n_resamples, dtype=float)
        degenerate = 0
        for index in range(self.n_resamples):
            labels = treatment[rng.permutation(treatment.size)]
            permuted = data.assign(**{spec.treatment_col: labels})
            replicate = evaluate_materialized(evaluator, permuted, spec, context)
            se, _, _ = _studentization_scale(replicate)
            effects[index] = replicate.point.effect
            scales[index] = se
            if se > 0.0 and math.isfinite(se):
                statistics[index] = replicate.point.effect / se
            else:
                statistics[index] = 0.0
                degenerate += 1
        return ResamplingDistribution(effects, scales, statistics, degenerate)

    def infer(
        self,
        evaluator: PointEvaluator,
        observed: PointEstimate,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
        *,
        materialized: MaterializedPoint | None = None,
    ) -> InferenceResult:
        if self.n_resamples < 99:
            raise ValueError("n_resamples must be at least 99.")
        working = _observed_materialization(
            evaluator, observed, data, spec, context, materialized
        )
        observed_se, observed_df, scale_source = _studentization_scale(working)
        observed_stat = observed.effect / observed_se if observed_se > 0.0 else math.nan
        distribution = self.replicate_distribution(evaluator, data, spec, context)
        statistics = distribution.statistics
        extreme = int(np.count_nonzero(np.abs(statistics) >= abs(observed_stat)))
        p_value = (extreme + 1.0) / (self.n_resamples + 1.0)
        critical = float(np.quantile(np.abs(statistics), 1.0 - spec.alpha, method="higher"))
        return InferenceResult(
            standard_error=observed_se,
            statistic=observed_stat,
            p_value=float(p_value),
            ci_low=float(observed.effect - critical * observed_se),
            ci_high=float(observed.effect + critical * observed_se),
            diagnostics={
                "degrees_of_freedom": observed_df,
                "n_resamples": self.n_resamples,
                "randomization_replays_full_point_evaluator": True,
                "assignment_replay": "fixed_size_label_permutation",
                "ci_method": "studentized_randomization_pivotal",
                "studentization": scale_source,
                "inference_target_col": working.spec.target_col,
                "degenerate_replicates": distribution.degenerate_replicates,
            },
        )


@dataclass(frozen=True)
class SharpNullRandomizationInference:
    """Перестановочный вывод для sharp null/постоянного аддитивного эффекта."""

    n_resamples: int = 999
    random_state: int = 42
    inference_id: str = "sharp_null_randomization"
    capabilities: frozenset[str] = frozenset({"design_aware", "full_refit", "sharp_null"})

    def infer(
        self,
        evaluator: PointEvaluator,
        observed: PointEstimate,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
        *,
        materialized: MaterializedPoint | None = None,
    ) -> InferenceResult:
        # Sharp-null вывод использует только переподогнанные точки, SE не нужен.
        del materialized
        if self.n_resamples < 99:
            raise ValueError("n_resamples must be at least 99.")
        treatment = np.asarray(data[spec.treatment_col]).copy()
        rng = np.random.default_rng(self.random_state)
        null_effects = np.empty(self.n_resamples, dtype=float)
        for index in range(self.n_resamples):
            labels = treatment[rng.permutation(treatment.size)]
            permuted = data.assign(**{spec.treatment_col: labels})
            null_effects[index] = evaluator.evaluate_point(
                permuted, spec, context
            ).effect
        extreme = int(np.count_nonzero(np.abs(null_effects) >= abs(observed.effect)))
        p_value = (extreme + 1.0) / (self.n_resamples + 1.0)
        standard_error = float(null_effects.std(ddof=1))
        lower_q, upper_q = np.quantile(
            null_effects,
            [spec.alpha / 2.0, 1.0 - spec.alpha / 2.0],
        )
        return InferenceResult(
            standard_error=standard_error,
            statistic=(
                observed.effect / standard_error if standard_error > 0.0 else math.nan
            ),
            p_value=float(p_value),
            ci_low=float(observed.effect - upper_q),
            ci_high=float(observed.effect - lower_q),
            diagnostics={
                "n_resamples": self.n_resamples,
                "null": "sharp_no_effect",
                "ci_assumption": "constant_additive_effect",
                "randomization_replays_full_point_evaluator": True,
                "assignment_replay": "fixed_size_label_permutation",
            },
        )


@dataclass(frozen=True)
class BootstrapTInference:
    """Bootstrap-t: pivot ``(effect* - effect) / SE*`` после полного refit реплики."""

    n_resamples: int = 999
    random_state: int = 42
    inference_id: str = "bootstrap_t"
    capabilities: frozenset[str] = frozenset({"full_refit", "superpopulation"})

    def replicate_distribution(
        self,
        evaluator: PointEvaluator,
        observed: PointEstimate,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> ResamplingDistribution:
        if self.n_resamples < 99:
            raise ValueError("n_resamples must be at least 99.")
        treatment = np.asarray(data[spec.treatment_col])
        control_index = np.flatnonzero(treatment == spec.control_value)
        treated_index = np.flatnonzero(treatment == spec.treatment_value)
        rng = np.random.default_rng(self.random_state)
        effects = np.empty(self.n_resamples, dtype=float)
        scales = np.empty(self.n_resamples, dtype=float)
        pivots = np.empty(self.n_resamples, dtype=float)
        degenerate = 0
        for index in range(self.n_resamples):
            sampled = np.concatenate(
                [
                    rng.choice(control_index, size=control_index.size, replace=True),
                    rng.choice(treated_index, size=treated_index.size, replace=True),
                ]
            )
            frame = data.iloc[sampled].reset_index(drop=True)
            replicate = evaluate_materialized(evaluator, frame, spec, context)
            se, _, _ = _studentization_scale(replicate)
            effects[index] = replicate.point.effect
            scales[index] = se
            if se > 0.0 and math.isfinite(se):
                pivots[index] = (replicate.point.effect - observed.effect) / se
            else:
                pivots[index] = 0.0
                degenerate += 1
        return ResamplingDistribution(effects, scales, pivots, degenerate)

    def infer(
        self,
        evaluator: PointEvaluator,
        observed: PointEstimate,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
        *,
        materialized: MaterializedPoint | None = None,
    ) -> InferenceResult:
        if self.n_resamples < 99:
            raise ValueError("n_resamples must be at least 99.")
        working = _observed_materialization(
            evaluator, observed, data, spec, context, materialized
        )
        observed_se, observed_df, scale_source = _studentization_scale(working)
        distribution = self.replicate_distribution(
            evaluator, observed, data, spec, context
        )
        pivots = distribution.statistics
        lower_q, upper_q = np.quantile(
            pivots,
            [spec.alpha / 2.0, 1.0 - spec.alpha / 2.0],
        )
        observed_stat = observed.effect / observed_se if observed_se > 0.0 else math.nan
        p_value = float(
            2.0
            * min(
                np.mean(pivots <= -abs(observed_stat)),
                np.mean(pivots >= abs(observed_stat)),
            )
        )
        return InferenceResult(
            standard_error=observed_se,
            statistic=observed_stat,
            p_value=min(1.0, p_value),
            ci_low=float(observed.effect - upper_q * observed_se),
            ci_high=float(observed.effect - lower_q * observed_se),
            diagnostics={
                "degrees_of_freedom": observed_df,
                "n_resamples": self.n_resamples,
                "bootstrap_replays_full_point_evaluator": True,
                "ci_method": "studentized_bootstrap_t",
                "studentization": scale_source,
                "inference_target_col": working.spec.target_col,
                "degenerate_replicates": distribution.degenerate_replicates,
            },
        )
