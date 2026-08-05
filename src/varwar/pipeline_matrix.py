from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping

from varwar.analysis import AnalysisMethod, PipelineStatus, RaceContext


@dataclass(frozen=True)
class PipelineComponents:
    adapter_id: str = "identity"
    structure_id: str = "none"
    assignment_id: str = "complete_randomization"
    target_id: str = "identity"
    analysis_method_id: str = "diff_means__welch"

    @property
    def pipeline_id(self) -> str:
        return (
            f"adapt.{self.adapter_id}"
            f"__structure.{self.structure_id}"
            f"__assign.{self.assignment_id}"
            f"__target.{self.target_id}"
            f"__analysis.{self.analysis_method_id}"
        )


@dataclass(frozen=True)
class PipelineMatrixRow:
    components: PipelineComponents
    context: RaceContext
    status: PipelineStatus
    reason: str = ""

    def to_record(self) -> dict[str, object]:
        return {
            "pipeline": self.components.pipeline_id,
            "adapter": self.components.adapter_id,
            "structure": self.components.structure_id,
            "assignment": self.components.assignment_id,
            "target_transformer": self.components.target_id,
            "analysis_method": self.components.analysis_method_id,
            "information_mode": self.context.information_mode.value,
            "temporal_access": self.context.temporal_access.value,
            "ranking_estimand_id": self.context.ranking_estimand_id,
            "status": self.status.value,
            "status_reason": self.reason,
        }


def build_pipeline_matrix(
    *,
    adapter_ids: Iterable[str],
    structure_ids: Iterable[str],
    assignment_ids: Iterable[str],
    target_ids: Iterable[str],
    analysis_methods: Mapping[str, AnalysisMethod],
    contexts: Iterable[RaceContext],
    structure_assignment_compatibility: Mapping[str, frozenset[str]] | None = None,
) -> tuple[PipelineMatrixRow, ...]:
    """Построить полное произведение и сохранить даже недоступные строки."""

    compatibility = structure_assignment_compatibility or {}
    rows: list[PipelineMatrixRow] = []
    for adapter, structure, assignment, target, method_id, context in product(
        tuple(adapter_ids),
        tuple(structure_ids),
        tuple(assignment_ids),
        tuple(target_ids),
        tuple(analysis_methods),
        tuple(contexts),
    ):
        components = PipelineComponents(
            adapter_id=adapter,
            structure_id=structure,
            assignment_id=assignment,
            target_id=target,
            analysis_method_id=method_id,
        )
        allowed_assignments = compatibility.get(structure)
        effective_context = RaceContext(
            assignment_id=assignment,
            information_mode=context.information_mode,
            temporal_access=context.temporal_access,
            ranking_estimand_id=context.ranking_estimand_id,
            panel_prior=context.panel_prior,
            replay_trace=context.replay_trace,
        )
        if allowed_assignments is not None and assignment not in allowed_assignments:
            rows.append(
                PipelineMatrixRow(
                    components,
                    effective_context,
                    PipelineStatus.UNSUPPORTED_ASSIGNMENT,
                    f"structure {structure!r} does not support {assignment!r}",
                )
            )
            continue
        status, reason = analysis_methods[method_id].compatibility(effective_context)
        rows.append(PipelineMatrixRow(components, effective_context, status, reason))
    ids = [
        (
            row.components.pipeline_id,
            row.context.information_mode,
            row.context.temporal_access,
            row.context.ranking_estimand_id,
        )
        for row in rows
    ]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Pipeline matrix contains duplicate context rows.")
    return tuple(rows)
