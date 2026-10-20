from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping

import pandas as pd

from varwar.analysis import (
    AnalysisMethod,
    DirectPointEvaluator,
    MaterializedPoint,
    PipelineStatus,
    PointEstimate,
    PointEstimator,
    RaceContext,
)
from varwar.core import ExperimentSpec
from varwar.pipeline_matrix import PipelineMatrixRow


TargetTransformerFactory = Callable[[], Any]


@dataclass
class TransformingPointEvaluator:
    """Повторяемая target-цепочка, пригодная для честного resampling."""

    point_estimator: PointEstimator
    transformer_factories: tuple[TargetTransformerFactory, ...] = ()
    pass_panel_prior: bool = False
    _last_diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def _fit_chain(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> tuple[pd.DataFrame, ExperimentSpec, dict[str, Any]]:
        working = data.copy(deep=False)
        current_spec = spec
        diagnostics: dict[str, Any] = {
            "target_stage_count": len(self.transformer_factories)
        }
        for index, factory in enumerate(self.transformer_factories, start=1):
            transformer = factory()
            adjusted = transformer.fit_transform(
                working,
                current_spec,
                context.panel_prior if self.pass_panel_prior else None,
            )
            column = f"__varwar_target_stage_{index}"
            working = working.copy(deep=False)
            working[column] = adjusted
            current_spec = current_spec.with_target(column)
            diagnostics[f"target_stage_{index}"] = transformer.name
            diagnostics.update(
                {
                    f"target_stage_{index}_{key}": value
                    for key, value in transformer.diagnostics().items()
                }
            )
        return working, current_spec, diagnostics

    def materialize(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> tuple[pd.DataFrame, ExperimentSpec]:
        """Рабочий frame с подогнанным target и spec, указывающий на него."""

        working, current_spec, _ = self._fit_chain(data, spec, context)
        return working, current_spec

    def evaluate_materialized(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> MaterializedPoint:
        """Одна подгонка цепочки: точка и frame/spec, на которых она посчитана."""

        working, current_spec, diagnostics = self._fit_chain(data, spec, context)
        point = self.point_estimator.estimate_point(working, current_spec, context)
        self._last_diagnostics = diagnostics
        estimate = PointEstimate(
            effect=point.effect,
            estimand_id=point.estimand_id,
            diagnostics={**point.diagnostics, **diagnostics},
            native_payload=point.native_payload,
        )
        return MaterializedPoint(estimate, working, current_spec)

    def evaluate_point(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        context: RaceContext,
    ) -> PointEstimate:
        return self.evaluate_materialized(data, spec, context).point

    def diagnostics(self) -> Mapping[str, Any]:
        return self._last_diagnostics


def execute_pipeline_row(
    row: PipelineMatrixRow,
    method: AnalysisMethod,
    data: pd.DataFrame,
    spec: ExperimentSpec,
    *,
    transformer_factories: tuple[TargetTransformerFactory, ...] = (),
    pass_panel_prior: bool = False,
) -> dict[str, Any]:
    """Выполнить строку матрицы, не теряя научную причину пропуска."""

    base = row.to_record()
    if row.status not in {PipelineStatus.ELIGIBLE, PipelineStatus.POINT_ONLY}:
        return base
    frame = row.context.analysis_frame(data, spec)
    evaluator = TransformingPointEvaluator(
        method.point_estimator,
        transformer_factories,
        pass_panel_prior,
    )
    try:
        if row.status == PipelineStatus.POINT_ONLY:
            point = evaluator.evaluate_point(frame, spec, row.context)
            treatment = frame[spec.treatment_col]
            return {
                **base,
                "status": PipelineStatus.POINT_ONLY.value,
                "effect": point.effect,
                "estimand_id": point.estimand_id,
                "standard_error": math.nan,
                "statistic": math.nan,
                "p_value": math.nan,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "n_control": int((treatment == spec.control_value).sum()),
                "n_treatment": int((treatment == spec.treatment_value).sum()),
                **{f"diag_{key}": value for key, value in point.diagnostics.items()},
            }
        result = method.analyze(
            frame,
            spec,
            row.context,
            point_evaluator=(
                evaluator
                if transformer_factories
                else DirectPointEvaluator(method.point_estimator)
            ),
        )
        return {
            **base,
            **result.to_record(),
            "status": PipelineStatus.RAN.value,
        }
    except Exception as exc:
        return {
            **base,
            "status": PipelineStatus.FAILED.value,
            "status_reason": f"{type(exc).__name__}: {exc}",
        }
