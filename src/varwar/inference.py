from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import stats

from varwar.analysis import (
    InferenceResult,
    PointEstimate,
    PointEvaluator,
    RaceContext,
)
from varwar.core import ExperimentSpec, require_columns


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
    ) -> InferenceResult:
        del evaluator, context
        control, treated = _groups(data, spec)
        se, df = _welch_parameters(control, treated)
        statistic = observed.effect / se if se > 0.0 else math.nan
        critical = float(stats.t.ppf(1.0 - spec.alpha / 2.0, df))
        return InferenceResult(
            standard_error=se,
            statistic=statistic,
            p_value=float(2.0 * stats.t.sf(abs(statistic), df)),
            ci_low=float(observed.effect - critical * se),
            ci_high=float(observed.effect + critical * se),
            diagnostics={"degrees_of_freedom": df, "inference_family": "welch"},
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
    ) -> InferenceResult:
        del evaluator, context
        control, treated = _groups(data, spec)
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
            diagnostics={"degrees_of_freedom": df, "inference_family": "pooled_student"},
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
    ) -> InferenceResult:
        del evaluator, context
        control, treated = _groups(data, spec)
        se, _ = _welch_parameters(control, treated)
        statistic = observed.effect / se if se > 0.0 else math.nan
        critical = float(stats.norm.ppf(1.0 - spec.alpha / 2.0))
        return InferenceResult(
            standard_error=se,
            statistic=statistic,
            p_value=float(2.0 * stats.norm.sf(abs(statistic))),
            ci_low=float(observed.effect - critical * se),
            ci_high=float(observed.effect + critical * se),
            diagnostics={"inference_family": "design_based_normal"},
        )


@dataclass(frozen=True)
class StudentizedRandomizationInference:
    n_resamples: int = 999
    random_state: int = 42
    inference_id: str = "welch_studentized_randomization"
    capabilities: frozenset[str] = frozenset({"design_aware", "full_refit"})

    def infer(
        self,
        evaluator: PointEvaluator,
        observed: PointEstimate,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> InferenceResult:
        if self.n_resamples < 99:
            raise ValueError("n_resamples must be at least 99.")
        control, treated = _groups(data, spec)
        observed_se, observed_df = _welch_parameters(control, treated)
        observed_stat = observed.effect / observed_se if observed_se > 0.0 else math.nan
        treatment = np.asarray(data[spec.treatment_col]).copy()
        rng = np.random.default_rng(self.random_state)
        statistics = np.empty(self.n_resamples, dtype=float)
        for index in range(self.n_resamples):
            permuted = data.copy(deep=False)
            labels = treatment[rng.permutation(treatment.size)]
            permuted = permuted.assign(**{spec.treatment_col: labels})
            point = evaluator.evaluate_point(permuted, spec, context)
            c, t = _groups(permuted, spec)
            se, _ = _welch_parameters(c, t)
            statistics[index] = point.effect / se if se > 0.0 else 0.0
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
    ) -> InferenceResult:
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
    n_resamples: int = 999
    random_state: int = 42
    inference_id: str = "bootstrap_t"
    capabilities: frozenset[str] = frozenset({"full_refit", "superpopulation"})

    def infer(
        self,
        evaluator: PointEvaluator,
        observed: PointEstimate,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> InferenceResult:
        if self.n_resamples < 99:
            raise ValueError("n_resamples must be at least 99.")
        treatment = np.asarray(data[spec.treatment_col])
        control_index = np.flatnonzero(treatment == spec.control_value)
        treated_index = np.flatnonzero(treatment == spec.treatment_value)
        control, treated = _groups(data, spec)
        observed_se, observed_df = _welch_parameters(control, treated)
        rng = np.random.default_rng(self.random_state)
        pivots = np.empty(self.n_resamples, dtype=float)
        for index in range(self.n_resamples):
            sampled = np.concatenate(
                [
                    rng.choice(control_index, size=control_index.size, replace=True),
                    rng.choice(treated_index, size=treated_index.size, replace=True),
                ]
            )
            replicate = data.iloc[sampled].reset_index(drop=True)
            point = evaluator.evaluate_point(replicate, spec, context)
            c, t = _groups(replicate, spec)
            se, _ = _welch_parameters(c, t)
            pivots[index] = (point.effect - observed.effect) / se if se > 0.0 else 0.0
        lower_q, upper_q = np.quantile(
            pivots,
            [spec.alpha / 2.0, 1.0 - spec.alpha / 2.0],
        )
        p_value = float(
            2.0
            * min(
                np.mean(pivots <= -abs(observed.effect / observed_se)),
                np.mean(pivots >= abs(observed.effect / observed_se)),
            )
        )
        return InferenceResult(
            standard_error=observed_se,
            statistic=observed.effect / observed_se,
            p_value=min(1.0, p_value),
            ci_low=float(observed.effect - upper_q * observed_se),
            ci_high=float(observed.effect - lower_q * observed_se),
            diagnostics={
                "degrees_of_freedom": observed_df,
                "n_resamples": self.n_resamples,
                "bootstrap_replays_full_point_evaluator": True,
                "ci_method": "studentized_bootstrap_t",
            },
        )
