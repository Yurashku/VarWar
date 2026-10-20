from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

import pandas as pd

from varwar.core import EstimateResult, ExperimentSpec


class TheoryGrade(str, Enum):
    """Сила теоретического обоснования зарегистрированного метода."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class CausalSafety(str, Enum):
    """Допустимость данных метода для причинного вывода в РКИ."""

    SAFE = "safe"
    CONDITIONAL = "conditional"
    POST_TREATMENT_RISK = "post_treatment_risk"
    NOT_CAUSAL = "not_causal"


class InformationMode(str, Enum):
    EXPERIMENT_ONLY = "experiment_only"
    PANEL_PRIOR = "panel_prior"


class TemporalAccess(str, Enum):
    PRE_TREATMENT_ONLY = "pre_treatment_only"
    FULL_TIMELINE = "full_timeline"


class PipelineStatus(str, Enum):
    ELIGIBLE = "eligible"
    RAN = "ran"
    POINT_ONLY = "point_only"
    UNAVAILABLE = "unavailable"
    ESTIMAND_MISMATCH = "estimand_mismatch"
    TEMPORAL_CONTRACT_VIOLATION = "temporal_contract_violation"
    INCOMPATIBLE_INFERENCE = "incompatible_inference"
    UNSUPPORTED_ASSIGNMENT = "unsupported_assignment"
    FAILED = "failed"


class InferenceQuality(str, Enum):
    GOOD = "good"
    WARNING = "warning"
    BAD = "bad"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class RaceContext:
    """Информационный контракт одной строки R&D-гонки.

    Контекст не является простой меткой: :meth:`analysis_frame` и
    :meth:`assignment_frame` создают представления, в которых отсутствуют
    запрещённые временным режимом столбцы.
    """

    assignment_id: str = "complete_randomization"
    information_mode: InformationMode = InformationMode.EXPERIMENT_ONLY
    temporal_access: TemporalAccess = TemporalAccess.PRE_TREATMENT_ONLY
    ranking_estimand_id: str = "sample_ate"
    panel_prior: Any | None = None
    replay_trace: Mapping[str, Any] = field(default_factory=dict)

    def analysis_frame(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
    ) -> pd.DataFrame:
        allowed = {
            spec.treatment_col,
            spec.target_col,
            spec.unit_col,
            spec.pre_target_col,
            spec.stratum_col,
            *spec.covariate_cols,
            *spec.pre_treatment_cols,
        }
        pre_features = set(spec.pre_treatment_features())
        allowed.update(pre_features)
        allowed.update(f"{column}__lag2" for column in pre_features)
        if self.temporal_access == TemporalAccess.FULL_TIMELINE:
            allowed.update(spec.expert_safe_in_time_cols)
            allowed.update(spec.gated_in_time_cols)
            allowed.update(spec.unsafe_demo_cols)
        columns = [column for column in data.columns if column in allowed]
        return data.loc[:, columns].copy(deep=False)

    def assignment_frame(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
    ) -> pd.DataFrame:
        allowed = {
            spec.unit_col,
            spec.stratum_col,
            *spec.covariate_cols,
            *spec.pre_treatment_cols,
        }
        allowed.discard(spec.target_col)
        allowed.discard(spec.treatment_col)
        columns = [column for column in data.columns if column in allowed]
        return data.loc[:, columns].copy(deep=False)


@dataclass(frozen=True)
class CompatibilityContract:
    """Машиночитаемые научные ограничения метода анализа."""

    estimand_id: str
    ranking_track: str
    theory_grade: TheoryGrade
    causal_safety: CausalSafety
    information_modes: frozenset[InformationMode] = frozenset(InformationMode)
    minimum_temporal_access: TemporalAccess = TemporalAccess.PRE_TREATMENT_ONLY
    point_assignment_ids: frozenset[str] = frozenset()
    assignment_ids: frozenset[str] = frozenset()
    inference_requirements: frozenset[str] = frozenset()
    inference_available: bool = True
    primary_ranking_eligible: bool = False

    def evaluate(self, context: RaceContext) -> tuple[PipelineStatus, str]:
        if context.information_mode not in self.information_modes:
            return PipelineStatus.UNAVAILABLE, "required information is unavailable"
        if (
            self.minimum_temporal_access == TemporalAccess.FULL_TIMELINE
            and context.temporal_access != TemporalAccess.FULL_TIMELINE
        ):
            return (
                PipelineStatus.TEMPORAL_CONTRACT_VIOLATION,
                "full timeline is required",
            )
        if context.ranking_estimand_id != self.estimand_id:
            return (
                PipelineStatus.ESTIMAND_MISMATCH,
                f"method targets {self.estimand_id}, ranking targets "
                f"{context.ranking_estimand_id}",
            )
        if (
            self.point_assignment_ids
            and context.assignment_id not in self.point_assignment_ids
        ):
            return (
                PipelineStatus.UNSUPPORTED_ASSIGNMENT,
                f"assignment {context.assignment_id!r} is unsupported",
            )
        if self.assignment_ids and context.assignment_id not in self.assignment_ids:
            return (
                PipelineStatus.POINT_ONLY,
                f"point is available but inference does not support "
                f"assignment {context.assignment_id!r}",
            )
        if not self.inference_available:
            return PipelineStatus.POINT_ONLY, "trusted inference is unavailable"
        return PipelineStatus.ELIGIBLE, ""


@dataclass(frozen=True)
class PointEstimate:
    effect: float
    estimand_id: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    native_payload: Any | None = None


@dataclass(frozen=True)
class InferenceResult:
    standard_error: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    analysis_method_id: str
    estimand_id: str
    effect: float
    standard_error: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    theory_grade: TheoryGrade
    causal_safety: CausalSafety
    inference_quality: InferenceQuality
    information_mode: InformationMode
    temporal_access: TemporalAccess
    n_control: int
    n_treatment: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = {
            "analysis_method": self.analysis_method_id,
            "estimand_id": self.estimand_id,
            "effect": self.effect,
            "standard_error": self.standard_error,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "theory_grade": self.theory_grade.value,
            "causal_safety": self.causal_safety.value,
            "inference_quality": self.inference_quality.value,
            "information_mode": self.information_mode.value,
            "temporal_access": self.temporal_access.value,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
        }
        record.update({f"diag_{key}": value for key, value in self.diagnostics.items()})
        return record


class PointEstimator(Protocol):
    method_id: str

    def estimate_point(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> PointEstimate: ...


@dataclass(frozen=True)
class MaterializedPoint:
    """Точка вместе с рабочим frame/spec, на которых она реально посчитана.

    Для target-цепочки ``data`` содержит подогнанный преобразованный target,
    а ``spec.target_col`` указывает на него. Любой non-native вывод обязан
    считать группы и SE именно здесь, а не на сыром исходе.
    """

    point: PointEstimate
    data: pd.DataFrame
    spec: ExperimentSpec


class PointEvaluator(Protocol):
    """Полная процедура получения точки, включая адаптивные преобразования."""

    def evaluate_point(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> PointEstimate: ...

    def materialize(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> tuple[pd.DataFrame, ExperimentSpec]: ...

    def evaluate_materialized(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> MaterializedPoint: ...


def evaluate_materialized(
    evaluator: PointEvaluator,
    data: pd.DataFrame,
    spec: ExperimentSpec,
    context: RaceContext,
) -> MaterializedPoint:
    """Посчитать точку и вернуть frame/spec той же подгонки target-цепочки.

    Если evaluator умеет делать это за один проход, цепочка подгоняется один
    раз. Evaluator без ``materialize`` отвергается: тихий откат к сырому
    target воспроизвёл бы ошибку SE на непреобразованном исходе.
    """

    combined = getattr(evaluator, "evaluate_materialized", None)
    if callable(combined):
        return combined(data, spec, context)
    materialize = getattr(evaluator, "materialize", None)
    if not callable(materialize):
        raise TypeError(
            f"{type(evaluator).__name__} must implement materialize(data, spec, "
            "context) so that inference uses the transformed target."
        )
    point = evaluator.evaluate_point(data, spec, context)
    frame, working_spec = materialize(data, spec, context)
    return MaterializedPoint(point, frame, working_spec)


@dataclass(frozen=True)
class DirectPointEvaluator:
    point_estimator: PointEstimator

    def evaluate_point(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> PointEstimate:
        return self.point_estimator.estimate_point(data, spec, context)

    def materialize(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> tuple[pd.DataFrame, ExperimentSpec]:
        del context
        return data, spec

    def evaluate_materialized(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> MaterializedPoint:
        return MaterializedPoint(self.evaluate_point(data, spec, context), data, spec)


class InferenceProcedure(Protocol):
    inference_id: str
    capabilities: frozenset[str]

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
        """``data``/``spec`` — исходные; ``materialized`` — та же подгонка,
        что дала ``observed`` (если не передана, процедура строит её сама)."""
        ...


@dataclass(frozen=True)
class AnalysisMethod:
    """Зарегистрированная и проверяемая пара оценки и статистического вывода."""

    method_id: str
    point_estimator: PointEstimator
    inference: InferenceProcedure
    contract: CompatibilityContract

    def compatibility(self, context: RaceContext) -> tuple[PipelineStatus, str]:
        status, reason = self.contract.evaluate(context)
        if status != PipelineStatus.ELIGIBLE:
            return status, reason
        missing = self.contract.inference_requirements.difference(
            self.inference.capabilities
        )
        if missing:
            return (
                PipelineStatus.INCOMPATIBLE_INFERENCE,
                f"inference lacks capabilities: {sorted(missing)}",
            )
        return PipelineStatus.ELIGIBLE, ""

    def analyze(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
        *,
        point_evaluator: PointEvaluator | None = None,
    ) -> AnalysisResult:
        status, reason = self.compatibility(context)
        if status != PipelineStatus.ELIGIBLE:
            raise ValueError(f"Analysis method is not eligible: {status.value}: {reason}")
        frame = context.analysis_frame(data, spec)
        evaluator = point_evaluator or DirectPointEvaluator(self.point_estimator)
        materialized = evaluate_materialized(evaluator, frame, spec, context)
        observed = materialized.point
        inference = self.inference.infer(
            evaluator,
            observed,
            frame,
            spec,
            context,
            materialized=materialized,
        )
        treatment = frame[spec.treatment_col]
        n_control = int((treatment == spec.control_value).sum())
        n_treatment = int((treatment == spec.treatment_value).sum())
        return AnalysisResult(
            analysis_method_id=self.method_id,
            estimand_id=observed.estimand_id,
            effect=observed.effect,
            standard_error=inference.standard_error,
            statistic=inference.statistic,
            p_value=inference.p_value,
            ci_low=inference.ci_low,
            ci_high=inference.ci_high,
            theory_grade=self.contract.theory_grade,
            causal_safety=self.contract.causal_safety,
            inference_quality=InferenceQuality.NOT_APPLICABLE,
            information_mode=context.information_mode,
            temporal_access=context.temporal_access,
            n_control=n_control,
            n_treatment=n_treatment,
            diagnostics={
                **observed.diagnostics,
                **inference.diagnostics,
                "inference_id": self.inference.inference_id,
                "ranking_track": self.contract.ranking_track,
            },
        )


@dataclass
class LegacyEstimatorPointAdapter:
    """Переходный point-adapter; канонический вывод берётся не отсюда."""

    estimator: Any
    estimand_id: str = "sample_ate"
    method_id: str = "legacy_estimator_point"

    def estimate_point(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> PointEstimate:
        del context
        result: EstimateResult = self.estimator.estimate(data, spec)
        return PointEstimate(
            effect=result.effect,
            estimand_id=self.estimand_id,
            diagnostics={"legacy_point_estimator": result.estimator},
            native_payload=result,
        )


@dataclass
class NativeEstimatorInference:
    """Переходный адаптер для неделимых библиотечных analysis methods."""

    estimator: Any
    inference_id: str = "native_estimator_inference"
    capabilities: frozenset[str] = frozenset({"native"})

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
        result = observed.native_payload
        if not isinstance(result, EstimateResult):
            if materialized is None:
                materialized = evaluate_materialized(evaluator, data, spec, context)
            result = self.estimator.estimate(materialized.data, materialized.spec)
        return InferenceResult(
            standard_error=result.standard_error,
            statistic=result.statistic,
            p_value=result.p_value,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            diagnostics={
                **dict(result.diagnostics),
                "native_inference": True,
            },
        )
