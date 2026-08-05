from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from varwar.analysis import (
    CausalSafety,
    InformationMode,
    PipelineStatus,
    RaceContext,
    TemporalAccess,
)
from varwar.analysis_registry import build_analysis_method_registry
from varwar.assignment import (
    CompleteRandomization,
    HypExAABestOfK,
    RerandomizationMahalanobis,
    RerandomizationMaxSMD,
    RerandomizationMixedBalance,
    StratifiedRandomization,
)
from varwar.core import ExperimentSpec
from varwar.feature_adapters import NumericFeatureBinner
from varwar.panel_prior import PanelPriorSummary
from varwar.pipeline_execution import execute_pipeline_row
from varwar.pipeline_matrix import PipelineComponents, PipelineMatrixRow
from varwar.randomization_structure import PrognosticScoreBlockBuilder
from varwar.synthetic_v3 import SyntheticWorld, generate_world, get_world_spec
from varwar.synthetic_v3 import list_worlds
from varwar.stratification import GeneticStrataReducer
from varwar.target_registry import TARGET_REGISTRY, TargetTransformDefinition
from varwar.validation import evaluate_inference_quality, summarize_trial_group


ASSIGNMENT_IDS = (
    "complete_randomization",
    "stratified_randomization",
    "rerandomization_mahalanobis",
    "rerandomization_max_smd",
    "rerandomization_mixed_balance",
    "hypex_aa_best_of_k",
)
STRUCTURE_IDS = ("none", "atomic_strata", "prognostic_blocks", "genetic_strata")
ADAPTER_IDS = ("identity", "numeric_binner")
RANKING_ESTIMANDS = (
    "sample_ate",
    "population_ate",
    "trimmed_mean_effect",
    "winsorized_mean_effect",
    "common_effect",
    "isolation_forest_trim_effect",
    "isolation_forest_winsorize_effect",
)


@dataclass(frozen=True)
class RNDRunConfig:
    """Воспроизводимая конфигурация одного этапа R&D."""

    stage: str = "smoke"
    world_ids: tuple[str, ...] = ("gaussian_r2_04_n200",)
    effects: tuple[float, ...] = (0.0, 0.2)
    n_trials: int = 5
    seed: int = 20260805
    resamples: int = 99
    include_hypex_assignment: bool = False
    pipeline_ids: tuple[str, ...] = ()
    max_executable_pipelines: int | None = None
    fast_only: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {"smoke", "screening", "confirmation", "power"}:
            raise ValueError("Unknown R&D stage.")
        if self.n_trials < 2:
            raise ValueError("At least two Monte Carlo trials are required.")
        if self.resamples < 99:
            raise ValueError("Resampling procedures need at least 99 replicates.")
        if 0.0 not in self.effects:
            raise ValueError("The effect grid must contain H0=0.")


@dataclass(frozen=True)
class RNDArtifacts:
    pipeline_matrix: pd.DataFrame
    trial_results: pd.DataFrame
    aggregates: pd.DataFrame
    leaders: pd.DataFrame
    output_dir: Path


def _memory_worker(config: RNDRunConfig, output_dir: str) -> None:
    run_rnd(config, output_dir)


def profile_rnd_peak_rss(
    config: RNDRunConfig,
    output_dir: str | Path,
    *,
    repeats: int = 3,
    poll_seconds: float = 0.02,
    cleanup_children: bool = True,
) -> pd.DataFrame:
    """Измерить peak RSS в отдельных процессах, а не через ``tracemalloc``."""

    if repeats < 1:
        raise ValueError("repeats must be positive.")
    import multiprocessing as mp
    import psutil
    import time

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(repeats):
        child_output = output / f"rss_repeat_{repeat + 1}"
        process = mp.get_context("spawn").Process(
            target=_memory_worker,
            args=(config, str(child_output)),
        )
        process.start()
        monitored = psutil.Process(process.pid)
        peak = 0
        started = perf_counter()
        while process.is_alive():
            try:
                rss = monitored.memory_info().rss
                rss += sum(
                    child.memory_info().rss for child in monitored.children(recursive=True)
                )
                peak = max(peak, int(rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(poll_seconds)
        process.join()
        rows.append(
            {
                "repeat": repeat + 1,
                "peak_rss_bytes": peak,
                "peak_rss_mb": peak / (1024.0 * 1024.0),
                "runtime_seconds": perf_counter() - started,
                "exit_code": process.exitcode,
            }
        )
        if cleanup_children and child_output.exists():
            import shutil

            shutil.rmtree(child_output)
    result = pd.DataFrame(rows)
    result.to_parquet(output / "memory_profile.parquet", index=False)
    return result


def profile_rnd_pipelines_peak_rss(
    config: RNDRunConfig,
    pipeline_ids: tuple[str, ...],
    output_dir: str | Path,
    *,
    repeats: int = 3,
) -> pd.DataFrame:
    """Измерить RSS каждого pipeline независимо и сохранить длинную таблицу."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for index, pipeline_id in enumerate(pipeline_ids, start=1):
        pipeline_config = replace(
            config,
            pipeline_ids=(pipeline_id,),
            max_executable_pipelines=None,
        )
        profile = profile_rnd_peak_rss(
            pipeline_config,
            output / f"pipeline_{index:03d}",
            repeats=repeats,
        )
        profile.insert(0, "pipeline", pipeline_id)
        frames.append(profile)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_parquet(output / "memory_profile.parquet", index=False)
    return result


def _spec(stratum_col: str | None = None) -> ExperimentSpec:
    return ExperimentSpec(
        treatment_col="treatment",
        target_col="y",
        unit_col="entity_id",
        covariate_cols=("x1", "x2", "x3", "x4", "region", "device"),
        pre_treatment_cols=("pre_target_lag1", "x1", "x2", "x3", "x4"),
        expert_safe_in_time_cols=("zone_b",),
        gated_in_time_cols=("zone_c",),
        unsafe_demo_cols=("post_outcome_proxy",),
        pre_target_col="pre_target_lag1",
        stratum_col=stratum_col,
    )


def _observed_frame(world: SyntheticWorld, labels: pd.Series) -> pd.DataFrame:
    treatment = np.asarray(labels, dtype=np.int8)
    cohort = world.experiment_cohort
    frame = world.assignment_data.copy(deep=False).copy()
    frame["treatment"] = treatment
    frame["y"] = np.where(treatment == 1, cohort["y1"], cohort["y0"])
    frame["zone_b"] = np.where(treatment == 1, cohort["zone_b1"], cohort["zone_b0"])
    frame["zone_c"] = np.where(treatment == 1, cohort["zone_c1"], cohort["zone_c0"])
    frame["post_outcome_proxy"] = np.where(
        treatment == 1,
        cohort["post_outcome_proxy1"],
        cohort["post_outcome_proxy0"],
    )
    frame["pre_target_lag1__lag2"] = frame["pre_target_lag2"]
    for column in ("x1", "x2", "x3", "x4"):
        frame[f"{column}__lag2"] = frame[column]
    return frame


def _panel_prior(world: SyntheticWorld) -> PanelPriorSummary:
    panel = world.pre_period_panel.sort_values(["entity_id", "report_date"]).copy()
    panel["pre_target_lag1"] = panel["pre_target"]
    panel["pre_target_lag1__lag2"] = panel.groupby("entity_id")["pre_target"].shift(1)
    for column in ("x1", "x2"):
        panel[f"{column}__lag2"] = panel.groupby("entity_id")[column].shift(1)
    panel = panel.dropna(subset=["pre_target_lag1__lag2", "x1__lag2", "x2__lag2"])
    cutoff = pd.Timestamp(panel["report_date"].max()) + pd.offsets.MonthBegin(1)
    return PanelPriorSummary.from_frame(
        panel,
        feature_cols=("pre_target_lag1__lag2", "x1__lag2", "x2__lag2"),
        target_col="pre_target_lag1",
        date_col="report_date",
        cutoff=cutoff,
        entity_col="entity_id",
        metadata={"synthetic_world": world.spec.world_id, "metric_version": "v3"},
    )


def _population_inputs(world: SyntheticWorld) -> tuple[dict[str, float], dict[str, float]]:
    strata = (
        world.finite_population["strata_a"].astype(str)
        + "|"
        + world.finite_population["strata_b"].astype(str)
    )
    weights = strata.value_counts(normalize=True).to_dict()
    moments = {
        column: float(pd.to_numeric(world.finite_population[column]).mean())
        for column in ("x1", "x2", "x3", "x4")
    }
    return {str(k): float(v) for k, v in weights.items()}, moments


def _mapped_population_weights(
    world: SyntheticWorld,
    design: pd.DataFrame,
    stratum_col: str | None,
) -> dict[str, float]:
    """Apply the fitted sample atomic-to-macro mapping to population shares."""

    if stratum_col is None or stratum_col not in design:
        return _population_inputs(world)[0]
    sample_raw = (
        world.assignment_data["strata_a"].astype(str)
        + "|"
        + world.assignment_data["strata_b"].astype(str)
    )
    mapping = pd.DataFrame(
        {"raw": sample_raw, "mapped": design[stratum_col].astype(str)}
    ).drop_duplicates("raw")
    raw_to_mapped = dict(zip(mapping["raw"], mapping["mapped"]))
    fallback = str(design[stratum_col].astype(str).value_counts().idxmax())
    population_raw = (
        world.finite_population["strata_a"].astype(str)
        + "|"
        + world.finite_population["strata_b"].astype(str)
    )
    mapped = population_raw.map(raw_to_mapped).fillna(fallback)
    shares = mapped.value_counts(normalize=True)
    return {str(key): float(value) for key, value in shares.items()}


def _target_status(
    target: TargetTransformDefinition,
    context: RaceContext,
) -> tuple[PipelineStatus, str]:
    if not target.available:
        return PipelineStatus.UNAVAILABLE, target.unavailable_reason
    if context.information_mode not in target.information_modes:
        return PipelineStatus.UNAVAILABLE, "target requires another information mode"
    if (
        target.minimum_temporal_access == TemporalAccess.FULL_TIMELINE
        and context.temporal_access != TemporalAccess.FULL_TIMELINE
    ):
        return PipelineStatus.TEMPORAL_CONTRACT_VIOLATION, "target requires full timeline"
    return PipelineStatus.ELIGIBLE, ""


def build_rnd_pipeline_matrix(
    analysis_methods: Mapping[str, Any],
) -> tuple[PipelineMatrixRow, ...]:
    """Полное произведение компонентов с явными причинами несовместимости."""

    rows: list[PipelineMatrixRow] = []
    for information in InformationMode:
        for temporal in TemporalAccess:
            for estimand in RANKING_ESTIMANDS:
                for adapter in ADAPTER_IDS:
                    for structure in STRUCTURE_IDS:
                      for assignment in ASSIGNMENT_IDS:
                        for target_id, target in TARGET_REGISTRY.items():
                          for method_id, method in analysis_methods.items():
                                context = RaceContext(
                                    assignment_id=assignment,
                                    information_mode=information,
                                    temporal_access=temporal,
                                    ranking_estimand_id=estimand,
                                )
                                components = PipelineComponents(
                                    adapter, structure, assignment, target_id, method_id
                                )
                                status, reason = _target_status(target, context)
                                if status == PipelineStatus.ELIGIBLE:
                                    status, reason = method.compatibility(context)
                                if status in {PipelineStatus.ELIGIBLE, PipelineStatus.POINT_ONLY}:
                                    if adapter == "numeric_binner" and structure not in {
                                        "atomic_strata", "genetic_strata"
                                    }:
                                        status, reason = (
                                            PipelineStatus.UNAVAILABLE,
                                            "numeric bins would be constructed but unused",
                                        )
                                    elif "stratified" in method_id and assignment != "stratified_randomization":
                                        status, reason = (
                                            PipelineStatus.UNSUPPORTED_ASSIGNMENT,
                                            "stratified inference requires within-stratum assignment in the canonical race",
                                        )
                                    elif "stratified" in method_id and structure == "none":
                                        status, reason = (
                                            PipelineStatus.UNAVAILABLE,
                                            "stratified analysis requires a constructed stratum",
                                        )
                                    elif (
                                        method_id.startswith("population_weighted_stratified")
                                        and structure != "atomic_strata"
                                    ):
                                        status, reason = (
                                            PipelineStatus.UNAVAILABLE,
                                            "population stratum shares are available for atomic_strata only",
                                        )
                                    elif (
                                        method_id.startswith("population_weighted_stratified")
                                        and adapter != "identity"
                                    ):
                                        status, reason = (
                                            PipelineStatus.UNAVAILABLE,
                                            "population weights require the same fitted adapter on the full population",
                                        )
                                    elif assignment == "stratified_randomization" and structure == "none":
                                        status, reason = (
                                            PipelineStatus.UNSUPPORTED_ASSIGNMENT,
                                            "stratified assignment requires a structure",
                                        )
                                    elif target_id != "identity" and not method_id.startswith("diff_means__"):
                                        status, reason = (
                                            PipelineStatus.INCOMPATIBLE_INFERENCE,
                                            "adaptive target transformation is only paired with full-refit mean-difference inference",
                                        )
                                    elif structure != "none" and assignment == "hypex_aa_best_of_k":
                                        status, reason = (
                                            PipelineStatus.UNSUPPORTED_ASSIGNMENT,
                                            "HypEx A/A policy does not consume VarWar structures",
                                        )
                                    elif (
                                        structure != "none"
                                        and assignment != "stratified_randomization"
                                        and "stratified" not in method_id
                                    ):
                                        status, reason = (
                                            PipelineStatus.UNSUPPORTED_ASSIGNMENT,
                                            "randomization structure would be constructed but unused",
                                        )
                                rows.append(PipelineMatrixRow(components, context, status, reason))
    return tuple(rows)


def _structure_frame(
    world: SyntheticWorld,
    structure_id: str,
    adapter_id: str,
) -> tuple[pd.DataFrame, ExperimentSpec]:
    def stabilize(values: pd.Series, minimum: int | None = None) -> pd.Series:
        if minimum is None:
            minimum = int(math.ceil(2.0 / min(world.spec.treatment_share, 1.0 - world.spec.treatment_share)))
        labels = values.astype("object").copy()
        counts = labels.value_counts(dropna=False)
        rare = set(counts[counts < minimum].index)
        if rare:
            labels.loc[labels.isin(rare)] = "__rare__"
        counts = labels.value_counts(dropna=False)
        if counts.get("__rare__", 0) < minimum and "__rare__" in counts:
            alternatives = counts.drop(labels=["__rare__"])
            if not alternatives.empty:
                labels.loc[labels == "__rare__"] = alternatives.idxmax()
        return labels.astype("category")

    frame = world.assignment_data.copy(deep=False).copy()
    category_cols = ["strata_a", "strata_b"]
    if adapter_id == "numeric_binner":
        binner = NumericFeatureBinner(columns=("x1", "x2"), n_bins=4)
        binner.fit(frame)
        frame = binner.augment(frame)
        category_cols.extend(("x1__bin", "x2__bin"))
    if structure_id == "none":
        return frame, _spec()
    if structure_id == "atomic_strata":
        frame["rnd_stratum"] = stabilize(
            frame[category_cols].astype(str).agg("|".join, axis=1)
        )
        return frame, _spec("rnd_stratum")
    spec = _spec("rnd_stratum")
    if structure_id == "genetic_strata":
        reducer = GeneticStrataReducer(
            category_cols=tuple(category_cols),
            reference_target_col="pre_target_lag1",
            n_strata=4,
            min_stratum_size=2,
            population_size=24,
            max_generations=40,
            elite_size=4,
            patience=10,
            random_state=42,
            output_col="rnd_stratum",
        )
        reducer.fit(frame)
        frame["rnd_stratum"] = stabilize(reducer.transform(frame))
        return frame, spec
    builder = PrognosticScoreBlockBuilder(
        feature_cols=("x1", "x2", "x3", "x4"),
        reference_target_col="pre_target_lag1",
        n_blocks=8,
        output_name="rnd_stratum",
    )
    builder.fit(frame, spec)
    frame["rnd_stratum"] = stabilize(builder.transform(frame))
    return frame, spec


def _assign(
    mechanism_id: str,
    data: pd.DataFrame,
    spec: ExperimentSpec,
    treatment_share: float,
    seed: int,
) -> Any:
    common = {"treatment_share": treatment_share}
    if mechanism_id == "complete_randomization":
        mechanism = CompleteRandomization(**common)
    elif mechanism_id == "stratified_randomization":
        mechanism = StratifiedRandomization(**common, min_per_arm=2)
    elif mechanism_id == "rerandomization_mahalanobis":
        mechanism = RerandomizationMahalanobis(
            **common, feature_cols=("x1", "x2", "x3", "x4"), n_pilot=64, max_attempts=200
        )
    elif mechanism_id == "rerandomization_max_smd":
        mechanism = RerandomizationMaxSMD(
            **common, feature_cols=("x1", "x2", "x3", "x4"), n_pilot=64, max_attempts=200
        )
    elif mechanism_id == "rerandomization_mixed_balance":
        mechanism = RerandomizationMixedBalance(
            **common,
            feature_cols=("x1", "x2", "region", "device"),
            n_pilot=64,
            max_attempts=200,
            anti_nuclear=True,
        )
    elif mechanism_id == "hypex_aa_best_of_k":
        mechanism = HypExAABestOfK(**common, n_iterations=10)
    else:
        raise KeyError(mechanism_id)
    return mechanism.assign(data, spec, np.random.default_rng(seed))


def _trial_truths(world: SyntheticWorld, estimand_id: str) -> dict[str, float]:
    native_map = {
        "trimmed_mean_effect": "trimmed_mean_10",
        "winsorized_mean_effect": "winsorized_mean_10",
    }
    native = world.native_truths.get(native_map.get(estimand_id, ""))
    native_value = native.sample_value if native else math.nan
    if estimand_id == "common_effect" and not world.spec.heterogeneous_effect:
        native_value = world.true_sample_ate
    return {
        "true_sample_ate": world.true_sample_ate,
        "true_population_ate": world.true_population_ate,
        "true_native_estimand": native_value,
    }


def _execution_rows(
    matrix: Iterable[PipelineMatrixRow],
    config: RNDRunConfig,
) -> list[PipelineMatrixRow]:
    rows = [row for row in matrix if row.status in {PipelineStatus.ELIGIBLE, PipelineStatus.POINT_ONLY}]
    if not config.include_hypex_assignment:
        rows = [row for row in rows if row.components.assignment_id != "hypex_aa_best_of_k"]
    if config.fast_only:
        slow_analysis = (
            "bootstrap",
            "randomization",
            "matching",
            "median_of_means",
            "catoni",
            "iforest",
        )
        rows = [
            row
            for row in rows
            if not any(part in row.components.analysis_method_id for part in slow_analysis)
            and "automl" not in row.components.target_id
        ]
    if config.pipeline_ids:
        selected = set(config.pipeline_ids)
        rows = [row for row in rows if row.components.pipeline_id in selected]
    if config.max_executable_pipelines is not None:
        limit = config.max_executable_pipelines
        selected: list[PipelineMatrixRow] = []
        uncovered = {
            "adapter": {row.components.adapter_id for row in rows},
            "analysis": {row.components.analysis_method_id for row in rows},
            "target": {row.components.target_id for row in rows},
            "assignment": {row.components.assignment_id for row in rows},
            "structure": {row.components.structure_id for row in rows},
            "information": {row.context.information_mode.value for row in rows},
            "temporal": {row.context.temporal_access.value for row in rows},
        }
        remaining = list(rows)
        while remaining and len(selected) < limit:
            def gain(row: PipelineMatrixRow) -> tuple[int, int]:
                values = (
                    ("adapter", row.components.adapter_id),
                    ("analysis", row.components.analysis_method_id),
                    ("target", row.components.target_id),
                    ("assignment", row.components.assignment_id),
                    ("structure", row.components.structure_id),
                    ("information", row.context.information_mode.value),
                    ("temporal", row.context.temporal_access.value),
                )
                return sum(value in uncovered[key] for key, value in values), -len(selected)
            best_index = max(range(len(remaining)), key=lambda index: gain(remaining[index]))
            chosen = remaining.pop(best_index)
            selected.append(chosen)
            uncovered["analysis"].discard(chosen.components.analysis_method_id)
            uncovered["adapter"].discard(chosen.components.adapter_id)
            uncovered["target"].discard(chosen.components.target_id)
            uncovered["assignment"].discard(chosen.components.assignment_id)
            uncovered["structure"].discard(chosen.components.structure_id)
            uncovered["information"].discard(chosen.context.information_mode.value)
            uncovered["temporal"].discard(chosen.context.temporal_access.value)
        rows = selected
    return rows


def run_rnd(
    config: RNDRunConfig,
    output_dir: str | Path,
) -> RNDArtifacts:
    """Запустить paired Monte Carlo и сохранить audit-friendly Parquet."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence(config.seed)
    prototype = generate_world(config.world_ids[0], seed=config.seed)
    population_weights, population_moments = _population_inputs(prototype)
    methods = build_analysis_method_registry(
        resamples=config.resamples,
        random_state=config.seed,
        population_weights=population_weights,
        population_moments=population_moments,
    )
    matrix = build_rnd_pipeline_matrix(methods)
    matrix_frame = pd.DataFrame(row.to_record() for row in matrix)
    matrix_frame.to_parquet(output / "pipeline_matrix.parquet", index=False)
    executable = _execution_rows(matrix, config)
    records: list[dict[str, Any]] = []
    world_trial_seeds = np.asarray(
        [
            int(child.generate_state(1, dtype=np.uint64)[0])
            for child in seed_sequence.spawn(len(config.world_ids) * config.n_trials)
        ],
        dtype=np.uint64,
    ).reshape(len(config.world_ids), config.n_trials)
    for world_index, world_id in enumerate(config.world_ids):
        for effect in config.effects:
            for trial in range(config.n_trials):
                # H0 and all H1 effect sizes share the same finite population,
                # cohort and assignment for a given Monte Carlo replication.
                seed = int(world_trial_seeds[world_index, trial])
                world_spec = replace(get_world_spec(world_id), effect=effect)
                world = generate_world(world_spec, seed=seed)
                prior = _panel_prior(world)
                _, moments = _population_inputs(world)
                grouped: dict[tuple[str, str, str], list[PipelineMatrixRow]] = {}
                for row in executable:
                    grouped.setdefault(
                        (
                            row.components.adapter_id,
                            row.components.structure_id,
                            row.components.assignment_id,
                        ), []
                    ).append(row)
                for (adapter_id, structure_id, assignment_id), rows in grouped.items():
                    design, spec = _structure_frame(world, structure_id, adapter_id)
                    group_methods = build_analysis_method_registry(
                        resamples=config.resamples,
                        random_state=seed,
                        population_weights=_mapped_population_weights(
                            world, design, spec.stratum_col
                        ),
                        population_moments=moments,
                    )
                    try:
                        assignment = _assign(
                            assignment_id, design, spec, world.spec.treatment_share, seed
                        )
                    except Exception as exc:
                        for row in rows:
                            records.append(
                                {
                                    **row.to_record(),
                                    "stage": config.stage,
                                    "world_id": world_id,
                                    "effect_size": effect,
                                    "hypothesis": "H0" if effect == 0.0 else "H1",
                                    "trial": trial,
                                    "seed": seed,
                                    "status": PipelineStatus.FAILED.value,
                                    "status_reason": f"assignment: {type(exc).__name__}: {exc}",
                                }
                            )
                        continue
                    observed = _observed_frame(world, assignment.labels)
                    if spec.stratum_col:
                        observed[spec.stratum_col] = design[spec.stratum_col]
                    for row in rows:
                        context = replace(
                            row.context,
                            panel_prior=(
                                prior
                                if row.context.information_mode == InformationMode.PANEL_PRIOR
                                else None
                            ),
                            replay_trace=(
                                assignment.replay_trace.to_record()
                                if assignment.replay_trace is not None
                                else {}
                            ),
                        )
                        effective = PipelineMatrixRow(row.components, context, row.status, row.reason)
                        target = TARGET_REGISTRY[row.components.target_id]
                        started = perf_counter()
                        result = execute_pipeline_row(
                            effective,
                            group_methods[row.components.analysis_method_id],
                            observed,
                            spec,
                            transformer_factories=(target.create,),
                            pass_panel_prior=bool(
                                target.metadata.get("requires_aggregate_prior", False)
                            ),
                        )
                        grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
                        safety_order = {
                            "safe": 0,
                            "conditional": 1,
                            "post_treatment_risk": 2,
                            "not_causal": 3,
                        }
                        analysis_grade = str(result.get("theory_grade", "D"))
                        analysis_safety = str(
                            result.get("causal_safety", "not_causal")
                        )
                        pipeline_grade = max(
                            (analysis_grade, target.theory_grade.value),
                            key=lambda value: grade_order[value],
                        )
                        pipeline_safety = max(
                            (analysis_safety, target.causal_safety.value),
                            key=lambda value: safety_order[value],
                        )
                        result.update(
                            {
                                "stage": config.stage,
                                "world_id": world_id,
                                "effect_size": effect,
                                "hypothesis": "H0" if effect == 0.0 else "H1",
                                "trial": trial,
                                "seed": seed,
                                "runtime_ms": 1000.0 * (perf_counter() - started),
                                "assignment_attempts": assignment.attempts,
                                "assignment_fallback": bool(
                                    assignment.replay_trace.fallback_used
                                    if assignment.replay_trace else False
                                ),
                                "target_theory_grade": target.theory_grade.value,
                                "target_causal_safety": target.causal_safety.value,
                                "analysis_theory_grade": analysis_grade,
                                "analysis_causal_safety": analysis_safety,
                                "theory_grade": pipeline_grade,
                                "causal_safety": pipeline_safety,
                                **_trial_truths(world, row.context.ranking_estimand_id),
                            }
                        )
                        records.append(result)
    trials = pd.DataFrame(records)
    trials.to_parquet(output / "trial_results.parquet", index=False)
    aggregates = aggregate_rnd_results(trials)
    aggregates.to_parquet(output / "aggregates.parquet", index=False)
    leaders = select_rnd_leaders(aggregates)
    leaders.to_parquet(output / "leaders.parquet", index=False)
    aggregates.to_csv(output / "aggregates.csv", index=False, encoding="utf-8-sig")
    leaders.to_csv(output / "leaders.csv", index=False, encoding="utf-8-sig")
    runnable_matrix = matrix_frame.loc[
        matrix_frame["status"].isin(
            [PipelineStatus.ELIGIBLE.value, PipelineStatus.POINT_ONLY.value]
        )
    ]
    from varwar.method_catalog import method_catalog

    methods_frame = pd.DataFrame(
        {
            "method_id": record.method_id,
            "category": record.category.value,
            "status": record.status.value,
            "summary": record.summary,
            "implementation": record.implementation,
            "race_scope": record.race_scope,
        }
        for record in method_catalog()
        if record.category.value != "CompositePipeline"
    )
    payload = {
        "aggregates": json.loads(aggregates.to_json(orient="records")),
        "leaders": json.loads(leaders.to_json(orient="records")),
        "pipelines": json.loads(runnable_matrix.to_json(orient="records")),
        "methods": json.loads(methods_frame.to_json(orient="records")),
        "worlds": json.loads(list_worlds().to_json(orient="records")),
    }
    (output / "excel_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "config": {
            **config.__dict__,
            "world_ids": list(config.world_ids),
            "effects": list(config.effects),
            "pipeline_ids": list(config.pipeline_ids),
        },
        "n_matrix_rows": len(matrix_frame),
        "n_executable_rows": len(executable),
        "n_trial_rows": len(trials),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from varwar.rnd_report import write_rnd_report_ru

    write_rnd_report_ru(
        output / "report_ru.md",
        aggregates,
        trials,
        matrix_frame,
        stage=config.stage,
    )
    return RNDArtifacts(matrix_frame, trials, aggregates, leaders, output)


def aggregate_rnd_results(trials: pd.DataFrame) -> pd.DataFrame:
    """Получить численные GT-метрики отдельно для каждого effect и H0/H1."""

    if trials.empty:
        return pd.DataFrame()
    successful = trials.loc[trials["status"] == PipelineStatus.RAN.value].copy()
    keys = [
        "pipeline",
        "world_id",
        "information_mode",
        "temporal_access",
        "ranking_estimand_id",
        "assignment",
        "structure",
        "target_transformer",
        "analysis_method",
    ]
    rows: list[dict[str, Any]] = []
    for key, frame in successful.groupby(keys, dropna=False, sort=False):
        h0 = frame.loc[frame["hypothesis"] == "H0"]
        alternatives = frame.loc[frame["hypothesis"] == "H1"]
        if len(h0) < 2 or len(alternatives) < 2:
            continue
        estimand = str(frame["ranking_estimand_id"].iloc[0])
        try:
            h0_summary = summarize_trial_group(h0, estimand_id=estimand)
            for effect, h1 in alternatives.groupby("effect_size", sort=True):
                h1_summary = summarize_trial_group(h1, estimand_id=estimand)
                grade = frame["theory_grade"].iloc[0]
                safety = frame["causal_safety"].iloc[0]
                quality = evaluate_inference_quality(
                    h0_summary,
                    h1_summary,
                    theory_grade=__import__("varwar.analysis", fromlist=["TheoryGrade"]).TheoryGrade(grade),
                    causal_safety=CausalSafety(safety),
                )
                record = dict(zip(keys, key))
                record.update({f"h0_{k}": v for k, v in h0_summary.items()})
                record.update({f"h1_{k}": v for k, v in h1_summary.items()})
                record.update(quality)
                record["raw_inference_quality"] = record["inference_quality"]
                record["scientific_sample_size_ok"] = bool(
                    h0_summary["n_trials"] >= 100 and h1_summary["n_trials"] >= 100
                )
                if (
                    not record["scientific_sample_size_ok"]
                    and record["inference_quality"] != "not_applicable"
                ):
                    # Smoke is an execution check.  Keep every numeric GT metric,
                    # but do not turn Monte Carlo noise into a scientific verdict.
                    record["inference_quality"] = "warning"
                record["effect_size"] = effect
                record["theory_grade"] = grade
                record["causal_safety"] = safety
                record["runtime_median_ms"] = float(frame["runtime_ms"].median())
                record["runtime_p95_ms"] = float(frame["runtime_ms"].quantile(0.95))
                rows.append(record)
        except (ValueError, KeyError):
            continue
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["power"] = result["h1_rejection_rate"]
    result["type_ii_error"] = 1.0 - result["power"]
    curve_keys = [
        "pipeline",
        "world_id",
        "information_mode",
        "temporal_access",
        "ranking_estimand_id",
    ]
    result["mde_80"] = math.nan
    for _, indices in result.groupby(curve_keys, dropna=False).groups.items():
        block = result.loc[indices].sort_values("effect_size")
        reached = block.loc[block["power"] >= 0.8, "effect_size"]
        if not reached.empty:
            result.loc[indices, "mde_80"] = float(reached.min())
    result["variance_reduction_vs_welch"] = math.nan
    comparison_keys = [
        "world_id",
        "information_mode",
        "temporal_access",
        "ranking_estimand_id",
        "assignment",
        "structure",
        "effect_size",
    ]
    for _, indices in result.groupby(comparison_keys, dropna=False).groups.items():
        block = result.loc[indices]
        baseline = block.loc[
            (block["target_transformer"] == "identity")
            & (block["analysis_method"] == "diff_means__welch"),
            "h1_empirical_sd",
        ]
        if baseline.empty or float(baseline.iloc[0]) <= 0.0:
            continue
        variance = float(baseline.iloc[0]) ** 2
        result.loc[indices, "variance_reduction_vs_welch"] = 1.0 - (
            result.loc[indices, "h1_empirical_sd"] ** 2 / variance
        )
    return result


def confirmation_pipeline_ids(
    screening: pd.DataFrame,
    *,
    max_candidates_per_track: int = 5,
) -> tuple[str, ...]:
    """Выбрать baseline и до пяти Парето-кандидатов каждого контура."""

    if screening.empty:
        return ()
    baselines = set(
        screening.loc[
            screening["analysis_method"].isin(
                {"diff_means__welch", "hypex_ab__integration"}
            ),
            "pipeline",
        ]
    )
    eligible = screening.loc[
        screening["scientific_sample_size_ok"]
        & screening["inference_quality"].isin(["good", "warning"])
    ].copy()
    groups = ["information_mode", "temporal_access", "ranking_estimand_id"]
    selected = set(baselines)
    for _, block in eligible.groupby(groups, dropna=False):
        block = block.sort_values(
            ["h1_rmse", "mde_80", "runtime_median_ms"],
            na_position="last",
        )
        selected.update(block["pipeline"].head(max_candidates_per_track))
    return tuple(sorted(selected))


def select_rnd_leaders(aggregates: pd.DataFrame) -> pd.DataFrame:
    """Ранжировать только внутри одного estimand и прошедшего экрана."""

    if aggregates.empty:
        return pd.DataFrame()
    eligible = aggregates.loc[
        (aggregates["inference_quality"] == "good")
        & aggregates["scientific_sample_size_ok"]
        & aggregates["h1_rmse"].notna()
    ].copy()
    if eligible.empty:
        return eligible
    group = ["world_id", "information_mode", "temporal_access", "ranking_estimand_id"]
    eligible["rank_rmse"] = eligible.groupby(group)["h1_rmse"].rank(method="min")
    eligible["rank_power"] = eligible.groupby(group)["h1_rejection_rate"].rank(
        method="min", ascending=False
    )
    eligible["pareto_score"] = eligible["rank_rmse"] + eligible["rank_power"]
    return eligible.sort_values([*group, "pareto_score", "runtime_median_ms"])
