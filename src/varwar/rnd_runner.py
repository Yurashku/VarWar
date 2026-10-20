from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
import tracemalloc
from typing import Any, Iterable, Mapping
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy import stats

from varwar.analysis import (
    CausalSafety,
    InformationMode,
    PipelineStatus,
    RaceContext,
    TemporalAccess,
    TheoryGrade,
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
from varwar.synthetic_v3 import (
    BASE_COVARIATE_COLS,
    BASE_PRE_TREATMENT_COLS,
    SyntheticWorld,
    SyntheticWorldSpec,
    generate_world,
    get_world_spec,
    has_heterogeneous_effect,
    spec_for_estimand,
    world_covariate_cols,
    world_pre_treatment_cols,
)
from varwar.synthetic_v3 import list_worlds
from varwar.stratification import GeneticStrataReducer
from varwar.target_registry import TARGET_REGISTRY, TargetTransformDefinition
from varwar.validation import (
    ValidationThresholds,
    benjamini_hochberg_reject,
    evaluate_inference_quality,
    summarize_trial_group,
)


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
    #: Execute a pipeline whose components ignore the information mode and
    #: temporal access once and copy the result to the other contexts.
    broadcast_mode_invariant: bool = True
    #: Trials ``0..k-1`` of every (world, effect) are run under ``tracemalloc``
    #: to measure the per-call peak of Python/NumPy allocations.
    memory_profile_trials: int = 2
    #: Forced effect-modifier centring for every estimand track.  ``None``
    #: (default) uses ``spec_for_estimand``: cohort centring for sample_ate,
    #: population centring for population_ate.
    effect_modifier_centering: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"smoke", "screening", "confirmation", "power"}:
            raise ValueError("Unknown R&D stage.")
        if self.n_trials < 2:
            raise ValueError("At least two Monte Carlo trials are required.")
        if self.resamples < 99:
            raise ValueError("Resampling procedures need at least 99 replicates.")
        if 0.0 not in self.effects:
            raise ValueError("The effect grid must contain H0=0.")
        if len(set(self.world_ids)) != len(self.world_ids):
            raise ValueError("world_ids must be unique.")
        if len(set(self.effects)) != len(self.effects):
            raise ValueError("effects must be unique.")
        if self.memory_profile_trials < 0:
            raise ValueError("memory_profile_trials must be non-negative.")

    def to_record(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "world_ids": list(self.world_ids),
            "effects": [float(value) for value in self.effects],
            "pipeline_ids": list(self.pipeline_ids),
        }


#: Configuration keys that must agree between shards of one stage.
SHARD_INVARIANT_CONFIG_KEYS = (
    "stage",
    "seed",
    "n_trials",
    "effects",
    "resamples",
    "include_hypex_assignment",
    "fast_only",
    "effect_modifier_centering",
    "broadcast_mode_invariant",
    "max_executable_pipelines",
)

SEED_COLUMNS = ("seed", "assignment_seed", "analysis_seed")

SEED_SCHEME = (
    "SeedSequence([config.seed, zlib.crc32(world_id), trial]).spawn(3) -> "
    "(world generation, assignment, analysis/resampling); identical for every "
    "effect size (CRN) and independent of shard composition and world order."
)


def trial_seeds(base_seed: int, world_id: str, trial: int) -> dict[str, int]:
    """Seeds of one Monte Carlo replication, independent of shard layout."""

    sequence = np.random.SeedSequence(
        [int(base_seed), zlib.crc32(world_id.encode("utf-8")), int(trial)]
    )
    world, assignment, analysis = sequence.spawn(3)
    return {
        "world": int(world.generate_state(1, dtype=np.uint64)[0]),
        "assignment": int(assignment.generate_state(1, dtype=np.uint64)[0]),
        "analysis": int(analysis.generate_state(1, dtype=np.uint64)[0]),
    }


def code_provenance() -> dict[str, Any]:
    """Version, git SHA and a hash of uncommitted changes under ``src``."""

    try:
        from importlib.metadata import version

        code_version = version("varwar")
    except Exception:  # pragma: no cover - editable install always has it
        code_version = "unknown"
    info: dict[str, Any] = {"code_version": code_version}
    root = Path(__file__).resolve().parents[2]
    try:
        def git(*args: str) -> bytes:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, timeout=30, check=True
            ).stdout

        info["git_sha"] = git("rev-parse", "HEAD").decode().strip()
        digest = hashlib.sha1(git("diff", "HEAD", "--", "src"))
        untracked = git("ls-files", "--others", "--exclude-standard", "--", "src")
        for name in sorted(untracked.decode().split()):
            path = root / name
            if path.is_file():
                digest.update(name.encode())
                digest.update(path.read_bytes())
        dirty = bool(git("status", "--porcelain", "--", "src").strip())
        info["git_src_dirty"] = dirty
        info["git_src_diff_sha1"] = digest.hexdigest() if dirty else ""
    except Exception:
        info.setdefault("git_sha", "unknown")
        info.setdefault("git_src_dirty", None)
        info.setdefault("git_src_diff_sha1", "")
    return info


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


def _spec(
    stratum_col: str | None = None,
    world_spec: SyntheticWorldSpec | None = None,
) -> ExperimentSpec:
    """Experiment spec; covariates come from the world declaration.

    Worlds without ``noise_in_covariates`` give the historical columns
    ``x1..x4, region, device`` / ``pre_target_lag1, x1..x4``.
    """

    covariates = (
        world_covariate_cols(world_spec) if world_spec is not None else BASE_COVARIATE_COLS
    )
    pre_treatment = (
        world_pre_treatment_cols(world_spec)
        if world_spec is not None
        else BASE_PRE_TREATMENT_COLS
    )
    return ExperimentSpec(
        treatment_col="treatment",
        target_col="y",
        unit_col="entity_id",
        covariate_cols=tuple(covariates),
        pre_treatment_cols=tuple(pre_treatment),
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
        return frame, _spec(None, world.spec)
    if structure_id == "atomic_strata":
        frame["rnd_stratum"] = stabilize(
            frame[category_cols].astype(str).agg("|".join, axis=1)
        )
        return frame, _spec("rnd_stratum", world.spec)
    spec = _spec("rnd_stratum", world.spec)
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
    if estimand_id == "common_effect" and not has_heterogeneous_effect(world.spec):
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
            # ~0.5 s per estimate; screening/confirmation include it.
            "gbm",
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



#: Requirement tags in ``TargetTransformDefinition.required_columns`` that
#: denote columns observed after assignment.
_IN_TIME_REQUIREMENTS = frozenset(
    {
        "expert_safe_in_time_features",
        "gated_in_time_features",
        "unsafe_demo_features",
        "in_time_features",
        "post_treatment_features",
    }
)


def _target_context_sensitivity(target: TargetTransformDefinition) -> tuple[bool, bool]:
    """Return ``(consumes_panel_prior, consumes_in_time)`` for a target.

    Explicit metadata keys ``consumes_panel_prior`` / ``consumes_in_time`` win.
    Otherwise the contract is derived from the registered requirements; a
    non-identity target without declared requirements is treated as
    context-sensitive (conservative).
    """

    metadata = dict(getattr(target, "metadata", {}) or {})
    required = set(getattr(target, "required_columns", ()) or ())
    panel = metadata.get("consumes_panel_prior")
    if panel is None:
        panel = bool(
            metadata.get("requires_aggregate_prior", False)
            or "panel_prior_summary" in required
            or frozenset(target.information_modes) == frozenset({InformationMode.PANEL_PRIOR})
        )
    in_time = metadata.get("consumes_in_time")
    if in_time is None:
        if target.method_id == "identity":
            in_time = False
        elif not required:
            in_time = True
        else:
            zones = tuple(metadata.get("zones", ()) or ())
            in_time = bool(
                target.minimum_temporal_access == TemporalAccess.FULL_TIMELINE
                or required & _IN_TIME_REQUIREMENTS
                or any(zone in {"B", "C"} for zone in zones)
            )
    return bool(panel), bool(in_time)


def _method_context_sensitivity(method: Any) -> tuple[bool, bool]:
    """Analysis methods read only spec columns that exist in every context.

    A method can opt in to context sensitivity through ``consumes_panel_prior``
    / ``consumes_in_time`` attributes on itself or its contract; a contract that
    requires the full timeline or only admits the panel mode is treated as
    sensitive to that axis.
    """

    contract = getattr(method, "contract", None)
    panel = getattr(method, "consumes_panel_prior", None)
    if panel is None:
        panel = getattr(contract, "consumes_panel_prior", None)
    if panel is None:
        modes = getattr(contract, "information_modes", frozenset(InformationMode))
        panel = frozenset(modes) == frozenset({InformationMode.PANEL_PRIOR})
    in_time = getattr(method, "consumes_in_time", None)
    if in_time is None:
        in_time = getattr(contract, "consumes_in_time", None)
    if in_time is None:
        in_time = (
            getattr(contract, "minimum_temporal_access", TemporalAccess.PRE_TREATMENT_ONLY)
            == TemporalAccess.FULL_TIMELINE
        )
    return bool(panel), bool(in_time)


def context_sensitivity(
    row: PipelineMatrixRow,
    methods: Mapping[str, Any],
) -> tuple[bool, bool]:
    target = TARGET_REGISTRY[row.components.target_id]
    target_panel, target_time = _target_context_sensitivity(target)
    method = methods.get(row.components.analysis_method_id)
    method_panel, method_time = _method_context_sensitivity(method)
    return target_panel or method_panel, target_time or method_time


def _context_label(information_mode: str, temporal_access: str) -> str:
    return f"{information_mode}|{temporal_access}"


def _sensitivity_label(panel: bool, in_time: bool) -> str:
    if panel and in_time:
        return "information+temporal"
    if panel:
        return "information"
    if in_time:
        return "temporal"
    return "none"


def _resolve_world_spec(config: RNDRunConfig, world_id: str, effect: float) -> Any:
    spec = replace(get_world_spec(world_id), effect=effect)
    if config.effect_modifier_centering is not None:
        if not hasattr(spec, "effect_modifier_centering"):
            raise ValueError(
                "This version of SyntheticWorldSpec has no effect_modifier_centering option."
            )
        spec = replace(spec, effect_modifier_centering=config.effect_modifier_centering)
    return spec


def _profiled_execute(
    profile_memory: bool,
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], float, float]:
    """Run ``function`` and return (result, runtime_ms, peak_bytes)."""

    if not profile_memory:
        started = perf_counter()
        result = function(*args, **kwargs)
        return result, 1000.0 * (perf_counter() - started), math.nan
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    started = perf_counter()
    try:
        result = function(*args, **kwargs)
    finally:
        runtime_ms = 1000.0 * (perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        if not was_tracing:
            tracemalloc.stop()
    return result, runtime_ms, float(max(peak - baseline, 0))


_MATRIX_CACHE: dict[tuple[Any, ...], tuple[tuple[PipelineMatrixRow, ...], pd.DataFrame]] = {}


def _registry_signature(methods: Mapping[str, Any]) -> tuple[Any, ...]:
    entries = []
    for method_id, method in sorted(methods.items()):
        inference = getattr(method, "inference", None)
        entries.append(
            (
                method_id,
                getattr(method, "contract", None),
                frozenset(getattr(inference, "capabilities", frozenset()) or ()),
            )
        )
    targets = tuple(
        (target_id, target.available, target.information_modes, target.minimum_temporal_access)
        for target_id, target in TARGET_REGISTRY.items()
    )
    return tuple(entries), targets


def _cached_pipeline_matrix(
    methods: Mapping[str, Any],
) -> tuple[tuple[PipelineMatrixRow, ...], pd.DataFrame]:
    """Build the 762k-row compatibility matrix once per registry signature.

    The matrix is a pure function of the method contracts, inference
    capabilities and the target registry, so a process-local cache keyed by
    that signature is exact.  It saves ~12 s per run in tests and sweeps.
    """

    try:
        key = _registry_signature(methods)
        hash(key)
    except TypeError:
        matrix = build_rnd_pipeline_matrix(methods)
        return matrix, pd.DataFrame(row.to_record() for row in matrix)
    cached = _MATRIX_CACHE.get(key)
    if cached is None:
        matrix = build_rnd_pipeline_matrix(methods)
        cached = (matrix, pd.DataFrame(row.to_record() for row in matrix))
        _MATRIX_CACHE.clear()
        _MATRIX_CACHE[key] = cached
    return cached


_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
_SAFETY_ORDER = {"safe": 0, "conditional": 1, "post_treatment_risk": 2, "not_causal": 3}


def run_rnd(
    config: RNDRunConfig,
    output_dir: str | Path,
) -> RNDArtifacts:
    """Запустить paired Monte Carlo и сохранить audit-friendly Parquet."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for world_id in config.world_ids:
        _resolve_world_spec(config, world_id, 0.0)
    prototype = generate_world(config.world_ids[0], seed=config.seed)
    population_weights, population_moments = _population_inputs(prototype)
    methods = build_analysis_method_registry(
        resamples=config.resamples,
        random_state=config.seed,
        population_weights=population_weights,
        population_moments=population_moments,
    )
    matrix, matrix_frame = _cached_pipeline_matrix(methods)
    matrix_frame.to_parquet(output / "pipeline_matrix.parquet", index=False)
    executable = _execution_rows(matrix, config)
    records: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[PipelineMatrixRow]] = {}
    for row in executable:
        grouped.setdefault(
            (
                row.components.adapter_id,
                row.components.structure_id,
                row.components.assignment_id,
            ),
            [],
        ).append(row)
    # Broadcast plan: one executed source row per invariance key.
    sensitivity: dict[tuple[str, str], tuple[bool, bool]] = {}
    plans: dict[tuple[str, str, str], list[tuple[PipelineMatrixRow, list[PipelineMatrixRow], tuple[bool, bool]]]] = {}
    for group_key, rows in grouped.items():
        by_key: dict[tuple[Any, ...], list[PipelineMatrixRow]] = {}
        for row in rows:
            component_key = (row.components.target_id, row.components.analysis_method_id)
            if component_key not in sensitivity:
                sensitivity[component_key] = context_sensitivity(row, methods)
            panel, in_time = sensitivity[component_key]
            info_key = (
                row.context.information_mode
                if panel or not config.broadcast_mode_invariant
                else None
            )
            time_key = (
                row.context.temporal_access
                if in_time or not config.broadcast_mode_invariant
                else None
            )
            key = (
                row.components.pipeline_id,
                row.context.ranking_estimand_id,
                info_key,
                time_key,
            )
            by_key.setdefault(key, []).append(row)
        plans[group_key] = [
            (
                members[0],
                members,
                sensitivity[(members[0].components.target_id, members[0].components.analysis_method_id)],
            )
            for members in by_key.values()
        ]
    n_executions = 0
    for world_id in config.world_ids:
        for effect in config.effects:
            world_spec = _resolve_world_spec(config, world_id, effect)
            for trial in range(config.n_trials):
                # H0 and all H1 effect sizes share the same finite population,
                # cohort and assignment for a given Monte Carlo replication;
                # the seeds depend on (seed, world_id, trial) only.
                seeds = trial_seeds(config.seed, world_id, trial)
                # One world per estimand-specific centring of the effect
                # modifier (cohort for sample_ate, population for
                # population_ate).  Centring only shifts y1 by a constant, so
                # every variant shares the same random streams, cohort,
                # covariates and therefore the same assignment.
                worlds: dict[Any, SyntheticWorld] = {}

                def world_for(estimand_id: str) -> SyntheticWorld:
                    variant = (
                        world_spec
                        if config.effect_modifier_centering is not None
                        else spec_for_estimand(world_spec, estimand_id)
                    )
                    if variant not in worlds:
                        worlds[variant] = generate_world(variant, seed=seeds["world"])
                    return worlds[variant]

                world = world_for("sample_ate")
                prior: PanelPriorSummary | None = None
                _, moments = _population_inputs(world)
                profile_memory = trial < config.memory_profile_trials
                seed_record = {
                    "seed": seeds["world"],
                    "assignment_seed": seeds["assignment"],
                    "analysis_seed": seeds["analysis"],
                }
                for (adapter_id, structure_id, assignment_id), plan in plans.items():
                    design, spec = _structure_frame(world, structure_id, adapter_id)
                    group_methods = build_analysis_method_registry(
                        resamples=config.resamples,
                        random_state=seeds["analysis"],
                        population_weights=_mapped_population_weights(
                            world, design, spec.stratum_col
                        ),
                        population_moments=moments,
                    )
                    try:
                        assignment = _assign(
                            assignment_id,
                            design,
                            spec,
                            world.spec.treatment_share,
                            seeds["assignment"],
                        )
                    except Exception as exc:
                        for _, members, (panel, in_time) in plan:
                            for row in members:
                                records.append(
                                    {
                                        **row.to_record(),
                                        "stage": config.stage,
                                        "world_id": world_id,
                                        "effect_size": effect,
                                        "hypothesis": "H0" if effect == 0.0 else "H1",
                                        "trial": trial,
                                        **seed_record,
                                        "status": PipelineStatus.FAILED.value,
                                        "status_reason": f"assignment: {type(exc).__name__}: {exc}",
                                        "context_sensitivity": _sensitivity_label(panel, in_time),
                                        "mode_invariant": not (panel or in_time),
                                        "broadcast": False,
                                        "executed_context": _context_label(
                                            row.context.information_mode.value,
                                            row.context.temporal_access.value,
                                        ),
                                    }
                                )
                        continue
                    observed_by_world: dict[int, pd.DataFrame] = {}

                    def observed_for(variant_world: SyntheticWorld) -> pd.DataFrame:
                        key = id(variant_world)
                        if key not in observed_by_world:
                            frame = _observed_frame(variant_world, assignment.labels)
                            if spec.stratum_col:
                                frame[spec.stratum_col] = design[spec.stratum_col]
                            observed_by_world[key] = frame
                        return observed_by_world[key]
                    replay = (
                        assignment.replay_trace.to_record()
                        if assignment.replay_trace is not None
                        else {}
                    )
                    for source, members, (panel, in_time) in plan:
                        if (
                            source.context.information_mode == InformationMode.PANEL_PRIOR
                            and prior is None
                        ):
                            prior = _panel_prior(world)
                        context = replace(
                            source.context,
                            panel_prior=(
                                prior
                                if source.context.information_mode == InformationMode.PANEL_PRIOR
                                else None
                            ),
                            replay_trace=replay,
                        )
                        effective = PipelineMatrixRow(
                            source.components, context, source.status, source.reason
                        )
                        target = TARGET_REGISTRY[source.components.target_id]
                        estimand_world = world_for(source.context.ranking_estimand_id)
                        result, runtime_ms, peak_bytes = _profiled_execute(
                            profile_memory,
                            execute_pipeline_row,
                            effective,
                            group_methods[source.components.analysis_method_id],
                            observed_for(estimand_world),
                            spec,
                            transformer_factories=(target.create,),
                            pass_panel_prior=bool(
                                target.metadata.get("requires_aggregate_prior", False)
                            ),
                        )
                        n_executions += 1
                        analysis_grade = str(result.get("theory_grade", "D"))
                        analysis_safety = str(result.get("causal_safety", "not_causal"))
                        pipeline_grade = max(
                            (analysis_grade, target.theory_grade.value),
                            key=lambda value: _GRADE_ORDER.get(value, 3),
                        )
                        pipeline_safety = max(
                            (analysis_safety, target.causal_safety.value),
                            key=lambda value: _SAFETY_ORDER.get(value, 3),
                        )
                        result.update(
                            {
                                "stage": config.stage,
                                "world_id": world_id,
                                "effect_size": effect,
                                "hypothesis": "H0" if effect == 0.0 else "H1",
                                "trial": trial,
                                **seed_record,
                                "runtime_ms": runtime_ms,
                                "memory_profiled": bool(profile_memory),
                                "peak_memory_bytes": peak_bytes,
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
                                "context_sensitivity": _sensitivity_label(panel, in_time),
                                "mode_invariant": not (panel or in_time),
                                "broadcast": False,
                                "executed_context": _context_label(
                                    source.context.information_mode.value,
                                    source.context.temporal_access.value,
                                ),
                                "effect_modifier_centering": getattr(
                                    estimand_world.spec, "effect_modifier_centering", None
                                ),
                                **_trial_truths(estimand_world, source.context.ranking_estimand_id),
                            }
                        )
                        records.append(result)
                        for row in members[1:]:
                            copy = dict(result)
                            copy.update(
                                {
                                    "information_mode": row.context.information_mode.value,
                                    "temporal_access": row.context.temporal_access.value,
                                    "broadcast": True,
                                }
                            )
                            records.append(copy)
    trials = pd.DataFrame(records)
    for column in SEED_COLUMNS:
        if column in trials:
            trials[column] = trials[column].astype("uint64")
    trials.to_parquet(output / "trial_results.parquet", index=False)
    aggregates = aggregate_rnd_results(trials)
    aggregates.to_parquet(output / "aggregates.parquet", index=False)
    leaders = select_rnd_leaders(aggregates)
    leaders.to_parquet(output / "leaders.parquet", index=False)
    aggregates.to_csv(output / "aggregates.csv", index=False, encoding="utf-8-sig")
    leaders.to_csv(output / "leaders.csv", index=False, encoding="utf-8-sig")
    finalists = select_confirmation_finalists(aggregates)
    write_stage_payload(output, aggregates, leaders, matrix_frame)
    metadata = {
        "config": config.to_record(),
        "provenance": code_provenance(),
        "seed_scheme": SEED_SCHEME,
        "n_matrix_rows": len(matrix_frame),
        "n_executable_rows": len(executable),
        "n_pipeline_executions": n_executions,
        "n_trial_rows": len(trials),
        "n_broadcast_rows": int(trials["broadcast"].sum()) if "broadcast" in trials else 0,
        "memory_metric": MEMORY_METRIC_NOTE,
        **finalists_metadata(finalists),
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


MEMORY_METRIC_NOTE = (
    "peak_memory_mb: per-call tracemalloc peak of Python/NumPy allocations inside "
    "execute_pipeline_row (estimate + inference), measured on the first "
    "memory_profile_trials trials of every (world, effect); it is not process RSS."
)


def write_stage_payload(
    output: Path,
    aggregates: pd.DataFrame,
    leaders: pd.DataFrame,
    matrix_frame: pd.DataFrame,
) -> None:
    """Write the JSON payload consumed by the Excel workbook builder."""

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


AGGREGATE_KEYS = [
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


def _first(frame: pd.DataFrame, column: str, default: Any) -> Any:
    if column not in frame or frame.empty:
        return default
    value = frame[column].iloc[0]
    return default if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _runtime_stats(frame: pd.DataFrame) -> tuple[float, float]:
    if "runtime_ms" not in frame:
        return math.nan, math.nan
    plain = frame
    if "memory_profiled" in frame:
        unprofiled = frame.loc[~frame["memory_profiled"].fillna(False).astype(bool)]
        if not unprofiled.empty:
            plain = unprofiled
    values = pd.to_numeric(plain["runtime_ms"], errors="coerce")
    return float(values.median()), float(values.quantile(0.95))


def _memory_stats(frame: pd.DataFrame) -> tuple[float, float]:
    if "peak_memory_bytes" not in frame:
        return math.nan, math.nan
    values = pd.to_numeric(frame["peak_memory_bytes"], errors="coerce").dropna()
    if values.empty:
        return math.nan, math.nan
    scale = 1024.0 * 1024.0
    return float(values.median() / scale), float(values.max() / scale)


def _analytic_mde(
    mean_se: float,
    sd_se: float,
    n_trials: float,
    df: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> tuple[float, float, float]:
    if not np.isfinite(mean_se):
        return math.nan, math.nan, math.nan
    if np.isfinite(df) and df > 0:
        multiplier = float(stats.t.ppf(1.0 - alpha / 2.0, df) + stats.t.ppf(power, df))
    else:
        multiplier = float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))
    mde = multiplier * mean_se
    if np.isfinite(sd_se) and n_trials and n_trials > 1:
        half = 1.96 * sd_se / math.sqrt(n_trials)
        return mde, multiplier * max(mean_se - half, 0.0), multiplier * (mean_se + half)
    return mde, math.nan, math.nan


def _inference_failure_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows whose inference diagnostics flag ``inference_failure``."""

    mask = pd.Series(False, index=frame.index)
    for column in frame.columns:
        if column == "inference_failure" or (
            column.startswith("diag_") and column.endswith("inference_failure")
        ):
            values = frame[column]
            flagged = values.map(lambda value: value is True or value == 1 or str(value).lower() == "true")
            mask |= flagged.fillna(False).astype(bool)
    return mask


def _superuniformity_test_id(
    base: Mapping[str, Any],
    estimand: str,
    sensitivity: str,
) -> tuple[Any, ...]:
    """Identify distinct H0 tests: contexts a pipeline ignores are one test."""

    information = (
        base["information_mode"] if sensitivity in {"information", "information+temporal", "unknown"} else "*"
    )
    temporal = (
        base["temporal_access"] if sensitivity in {"temporal", "information+temporal", "unknown"} else "*"
    )
    return (base["pipeline"], base["world_id"], estimand, information, temporal)


def aggregate_rnd_results(
    trials: pd.DataFrame,
    *,
    thresholds: ValidationThresholds = ValidationThresholds(),
) -> pd.DataFrame:
    """Получить численные GT-метрики отдельно для каждого effect и H0/H1.

    A failure in one effect size does not discard H0 or the other effects.
    Groups with execution failures are reported with ``failure_rate``; a group
    whose H0 or H1 cannot be summarised because every trial failed becomes a
    ``bad`` row with reason ``execution_failure``.  The p-value
    super-uniformity decision is made with Benjamini-Hochberg across all
    distinct H0 tests of the call (one stage, or one merged stage).
    """

    if trials.empty:
        return pd.DataFrame()
    ran_value = PipelineStatus.RAN.value
    failed_value = PipelineStatus.FAILED.value
    pending: list[dict[str, Any]] = []
    skipped: list[str] = []
    for key, frame in trials.groupby(AGGREGATE_KEYS, dropna=False, sort=False):
        base = dict(zip(AGGREGATE_KEYS, key))
        estimand = str(base["ranking_estimand_id"])
        status = frame["status"].astype(str)
        hypothesis = frame["hypothesis"].astype(str)
        inference_failed = _inference_failure_mask(frame) & (status == ran_value)
        h0_all = frame.loc[hypothesis == "H0"]
        h0 = h0_all.loc[status.loc[h0_all.index] == ran_value]
        alternatives = frame.loc[hypothesis == "H1"]
        if alternatives.empty:
            continue
        h0_summary: dict[str, Any] | None = None
        h0_error = ""
        if len(h0) >= 2:
            try:
                h0_summary = summarize_trial_group(h0, estimand_id=estimand)
            except (ValueError, KeyError) as exc:
                h0_error = f"H0 summary: {type(exc).__name__}: {exc}"
        else:
            h0_error = "H0: fewer than two successful trials"
        executed_context = str(
            _first(
                frame,
                "executed_context",
                _context_label(str(base["information_mode"]), str(base["temporal_access"])),
            )
        )
        grade = str(_first(frame.loc[status == ran_value], "theory_grade", _first(frame, "theory_grade", "D")))
        safety = str(
            _first(frame.loc[status == ran_value], "causal_safety", _first(frame, "causal_safety", "not_causal"))
        )
        runtime_median, runtime_p95 = _runtime_stats(frame.loc[status == ran_value])
        memory_median, memory_max = _memory_stats(frame.loc[status == ran_value])
        for effect, h1_all in alternatives.groupby("effect_size", sort=True):
            h1 = h1_all.loc[status.loc[h1_all.index] == ran_value]
            n_total = len(h0_all) + len(h1_all)
            n_failed = int((status.loc[h0_all.index] == failed_value).sum()) + int(
                (status.loc[h1_all.index] == failed_value).sum()
            )
            n_inference_failed = int(inference_failed.loc[h0_all.index].sum()) + int(
                inference_failed.loc[h1_all.index].sum()
            )
            n_point_only = int(
                (status.loc[h0_all.index] == PipelineStatus.POINT_ONLY.value).sum()
                + (status.loc[h1_all.index] == PipelineStatus.POINT_ONLY.value).sum()
            )
            record: dict[str, Any] = {
                **base,
                "effect_size": float(effect),
                "theory_grade": grade,
                "causal_safety": safety,
                "mode_invariant": bool(_first(frame, "mode_invariant", False)),
                "context_sensitivity": str(_first(frame, "context_sensitivity", "unknown")),
                "executed_context": executed_context,
                "n_trial_rows": n_total,
                "n_failed_trials": n_failed,
                "n_inference_failed_trials": n_inference_failed,
                # Execution failures plus ran trials whose inference reported
                # inference_failure=True (e.g. >5% failed bootstrap replicates).
                "failure_rate": (n_failed + n_inference_failed) / n_total if n_total else math.nan,
                "execution_failure_rate": n_failed / n_total if n_total else math.nan,
                "inference_failure_rate": n_inference_failed / n_total if n_total else math.nan,
                "runtime_median_ms": runtime_median,
                "runtime_p95_ms": runtime_p95,
                "peak_memory_mb": memory_median,
                "peak_memory_mb_max": memory_max,
            }
            h1_summary: dict[str, Any] | None = None
            error = h0_error
            if h0_summary is not None:
                if len(h1) >= 2:
                    try:
                        h1_summary = summarize_trial_group(h1, estimand_id=estimand)
                    except (ValueError, KeyError) as exc:
                        error = f"H1 summary: {type(exc).__name__}: {exc}"
                else:
                    error = "H1: fewer than two successful trials"
            if h0_summary is None or h1_summary is None:
                if n_point_only == n_total:
                    continue  # point-only rows have no inference to validate
                label = f"{base['pipeline']} | {base['world_id']} | {executed_context} | effect={effect}: {error}"
                skipped.append(label)
                if n_failed > 0:
                    record.update(
                        {
                            "inference_quality": "bad",
                            "raw_inference_quality": "bad",
                            "verdict_reasons": "execution_failure",
                            "aggregation_error": error,
                            "scientific_sample_size_ok": False,
                        }
                    )
                    pending.append({"record": record})
                continue
            record["h0_n_trials_total"] = len(h0_all)
            record["h1_n_trials_total"] = len(h1_all)
            pending.append(
                {
                    "record": record,
                    "h0": h0_summary,
                    "h1": h1_summary,
                    "test_id": _superuniformity_test_id(base, estimand, record["context_sensitivity"]),
                }
            )
    if skipped:
        warnings.warn(
            f"{len(skipped)} aggregate group(s) could not be summarised; first: {skipped[0]}",
            RuntimeWarning,
            stacklevel=2,
        )
    # Stage-level Benjamini-Hochberg over distinct H0 super-uniformity tests.
    test_p: dict[tuple[Any, ...], float] = {}
    for item in pending:
        if "h0" in item and bool(item["h0"].get("truth_is_null", True)):
            test_p.setdefault(item["test_id"], float(item["h0"].get("pvalue_superuniform_p", math.nan)))
    test_ids = list(test_p)
    bh = benjamini_hochberg_reject(
        [test_p[test] for test in test_ids], q=thresholds.superuniformity_fdr_q
    )
    rejected = dict(zip(test_ids, bh.tolist()))
    rows: list[dict[str, Any]] = []
    for item in pending:
        record = item["record"]
        if "h0" not in item:
            rows.append(record)
            continue
        h0_summary, h1_summary = item["h0"], item["h1"]
        nan_rate = max(
            float(h0_summary.get("nan_pvalue_rate", 0.0) or 0.0),
            float(h1_summary.get("nan_pvalue_rate", 0.0) or 0.0),
        )
        quality = evaluate_inference_quality(
            h0_summary,
            h1_summary,
            theory_grade=TheoryGrade(record["theory_grade"]),
            causal_safety=CausalSafety(record["causal_safety"]),
            thresholds=thresholds,
            superuniformity_rejected=rejected.get(item["test_id"]),
            failure_rate=float(record["execution_failure_rate"]),
            inference_failure_rate=float(record["inference_failure_rate"]),
            nan_pvalue_rate=nan_rate,
        )
        record.update({f"h0_{k}": v for k, v in h0_summary.items()})
        record.update({f"h1_{k}": v for k, v in h1_summary.items()})
        record["h0_pvalue_superuniform_bh_reject"] = rejected.get(item["test_id"], False)
        record["nan_pvalue_rate"] = nan_rate
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
            record["verdict_reasons"] = ";".join(
                part for part in ("mc_trials_lt_100", record.get("verdict_reasons", "")) if part
            )
        mde, mde_low, mde_high = _analytic_mde(
            float(h1_summary.get("mean_standard_error", math.nan)),
            float(h1_summary.get("sd_standard_error", math.nan)),
            float(h1_summary.get("n_trials", math.nan)),
            float(h1_summary.get("df_median", math.nan)),
            alpha=thresholds.alpha,
        )
        record["mde_80"] = mde
        record["mde_80_ci_low"] = mde_low
        record["mde_80_ci_high"] = mde_high
        rows.append(record)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result.attrs["skipped_groups"] = skipped
    if "h1_rejection_rate" not in result:
        result["h1_rejection_rate"] = math.nan
    if "h1_rmse" not in result:
        result["h1_rmse"] = math.nan
    if "h1_empirical_sd" not in result:
        result["h1_empirical_sd"] = math.nan
    result["power"] = result["h1_rejection_rate"]
    result["type_ii_error"] = 1.0 - result["power"]
    curve_keys = [
        "pipeline",
        "world_id",
        "information_mode",
        "temporal_access",
        "ranking_estimand_id",
    ]
    result["mde_80_grid"] = math.nan
    for _, indices in result.groupby(curve_keys, dropna=False).groups.items():
        block = result.loc[indices]
        reached = block.loc[block["power"] >= 0.8, "effect_size"]
        if not reached.empty:
            result.loc[indices, "mde_80_grid"] = float(reached.min())
    _attach_welch_baselines(result)
    return result


def _is_welch_baseline(frame: pd.DataFrame, *, complete_randomization: bool) -> pd.Series:
    mask = (frame["target_transformer"] == "identity") & (
        frame["analysis_method"] == "diff_means__welch"
    )
    if complete_randomization:
        mask &= frame["assignment"] == "complete_randomization"
        mask &= frame["structure"] == "none"
        if "adapter" in frame:
            mask &= frame["adapter"] == "identity"
        else:
            mask &= frame["pipeline"].astype(str).str.startswith("adapt.identity__")
    return mask


def _attach_welch_baselines(result: pd.DataFrame) -> None:
    """Variance reduction within the same design and against CR + Welch."""

    result["variance_reduction_vs_welch"] = math.nan
    result["variance_reduction_vs_welch_cr"] = math.nan
    result["rmse_ratio_vs_welch_cr"] = math.nan
    same_design = [
        "world_id",
        "information_mode",
        "temporal_access",
        "ranking_estimand_id",
        "assignment",
        "structure",
        "effect_size",
    ]
    welch = result.loc[_is_welch_baseline(result, complete_randomization=False)]
    if not welch.empty:
        baseline = (
            welch.groupby(same_design, dropna=False)["h1_empirical_sd"].first().rename("_base_sd")
        )
        merged = result[same_design].merge(
            baseline.reset_index(), on=same_design, how="left"
        )
        base_sd = merged["_base_sd"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = 1.0 - result["h1_empirical_sd"].to_numpy(float) ** 2 / base_sd**2
        values[~(base_sd > 0.0)] = math.nan
        result["variance_reduction_vs_welch"] = values
    # Cross-design baseline: complete randomisation + identity + Welch in the
    # same world, context and effect (any estimand track: Welch is SATE-only).
    cross = ["world_id", "information_mode", "temporal_access", "effect_size"]
    welch_cr = result.loc[_is_welch_baseline(result, complete_randomization=True)]
    if welch_cr.empty:
        return
    baseline = welch_cr.groupby(cross, dropna=False).agg(
        _base_sd=("h1_empirical_sd", "first"), _base_rmse=("h1_rmse", "first")
    )
    merged = result[cross].merge(baseline.reset_index(), on=cross, how="left")
    base_sd = merged["_base_sd"].to_numpy(float)
    base_rmse = merged["_base_rmse"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = 1.0 - result["h1_empirical_sd"].to_numpy(float) ** 2 / base_sd**2
        ratio = result["h1_rmse"].to_numpy(float) / base_rmse
    vr[~(base_sd > 0.0)] = math.nan
    ratio[~(base_rmse > 0.0)] = math.nan
    result["variance_reduction_vs_welch_cr"] = vr
    result["rmse_ratio_vs_welch_cr"] = ratio


CONFIRMATION_SELECTION_RULE = (
    "Отбор финалистов confirmation (кодом, select_confirmation_finalists). "
    "Контур = (information_mode, temporal_access, ranking_estimand_id). "
    "В контуре учитываются только строки с scientific_sample_size_ok. "
    "Пайплайн исключается из контура, если хотя бы одна его строка "
    "(любой мир, любой эффект) имеет inference_quality='bad', "
    "failure_rate>0 или causal_safety вне {{safe, conditional}}. "
    "Оставшиеся упорядочиваются по среднему log(RMSE/RMSE_Welch_CR) по мирам "
    "и эффектам (RMSE_Welch_CR — identity+Welch при полной рандомизации в том же "
    "мире, контексте и эффекте; если базовой строки нет, используется средний "
    "ранг RMSE внутри ячейки), затем по медиане runtime. "
    "Берутся первые k={k} пайплайнов каждого контура. Всегда добавляются "
    "якоря: identity+Welch и HypEx A/B при полной рандомизации."
)


def select_confirmation_finalists(
    screening: pd.DataFrame,
    *,
    max_candidates_per_track: int = 5,
) -> pd.DataFrame:
    """Ranked finalists per track with the score that produced the ranking."""

    columns = ["track", "pipeline", "rank", "score", "n_rows", "n_worlds", "n_worlds_in_track", "role"]
    if screening.empty or "pipeline" not in screening:
        return pd.DataFrame(columns=columns)
    frame = screening.copy()
    for column, default in (
        ("scientific_sample_size_ok", False),
        ("failure_rate", 0.0),
        ("causal_safety", "safe"),
        ("rmse_ratio_vs_welch_cr", math.nan),
        ("runtime_median_ms", math.nan),
    ):
        if column not in frame:
            frame[column] = default
    # Исключение считается по ВСЕМ строкам пайплайна в контуре (в т.ч. строкам полного
    # отказа, у которых scientific_sample_size_ok=False); фильтр по объёму MC
    # применяется только к строкам, по которым считается score.
    track = ["information_mode", "temporal_access", "ranking_estimand_id"]
    excluded: set[tuple[Any, ...]] = set()
    for key, rows_block in frame.groupby(track + ["pipeline"], dropna=False):
        if (
            rows_block["inference_quality"].eq("bad").any()
            or (pd.to_numeric(rows_block["failure_rate"], errors="coerce").fillna(0.0) > 0).any()
            or ~rows_block["causal_safety"].isin(["safe", "conditional"]).all()
        ):
            excluded.add(tuple(key))
    track_worlds = frame.groupby(track, dropna=False)["world_id"].nunique()
    frame = frame.loc[frame["scientific_sample_size_ok"].fillna(False).astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    anchors = screening.loc[
        screening["analysis_method"].isin({"diff_means__welch", "hypex_ab__integration"})
        & (screening["target_transformer"] == "identity")
        & (screening.get("assignment", "complete_randomization") == "complete_randomization")
        & (screening.get("structure", "none") == "none"),
        "pipeline",
    ].drop_duplicates()
    for pipeline in sorted(anchors):
        rows.append(
            {"track": "anchor", "pipeline": pipeline, "rank": 0, "score": math.nan,
             "n_rows": int((screening["pipeline"] == pipeline).sum()),
             "n_worlds": int(screening.loc[screening["pipeline"] == pipeline, "world_id"].nunique()),
             "role": "anchor"}
        )
    if frame.empty:
        return pd.DataFrame(rows, columns=columns)
    cell = ["world_id", "information_mode", "temporal_access", "ranking_estimand_id", "effect_size"]
    frame["_cell_rank"] = frame.groupby(cell, dropna=False)["h1_rmse"].rank(pct=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["_log_ratio"] = np.log(pd.to_numeric(frame["rmse_ratio_vs_welch_cr"], errors="coerce"))
    for track_key, block in frame.groupby(track, dropna=False):
        label = "|".join(str(value) for value in track_key)
        summary = []
        for pipeline, rows_block in block.groupby("pipeline"):
            key = tuple(track_key) + (pipeline,) if isinstance(track_key, tuple) else (track_key, pipeline)
            if key in excluded:
                continue
            log_ratio = rows_block["_log_ratio"]
            if log_ratio.notna().all():
                score = float(log_ratio.mean())
                kind = 0
            else:
                score = float(rows_block["_cell_rank"].mean())
                kind = 1
            summary.append(
                (kind, score, float(rows_block["runtime_median_ms"].median()), pipeline,
                 len(rows_block), int(rows_block["world_id"].nunique()))
            )
        summary.sort(key=lambda item: (item[0], item[1], item[2] if np.isfinite(item[2]) else math.inf, item[3]))
        for rank, (kind, score, _, pipeline, n_rows, n_worlds) in enumerate(
            summary[:max_candidates_per_track], start=1
        ):
            rows.append(
                {"track": label, "pipeline": pipeline, "rank": rank, "score": score,
                 "n_rows": n_rows, "n_worlds": n_worlds,
                 "n_worlds_in_track": int(track_worlds.get(track_key, n_worlds)),
                 "role": "finalist" if kind == 0 else "finalist_rank_fallback"}
            )
    return pd.DataFrame(rows, columns=columns)


def finalists_metadata(
    finalists: pd.DataFrame,
    *,
    max_candidates_per_track: int = 5,
) -> dict[str, Any]:
    return {
        "finalists": sorted(set(finalists["pipeline"])) if not finalists.empty else [],
        "finalists_by_track": json.loads(finalists.to_json(orient="records")),
        "selection_rule": CONFIRMATION_SELECTION_RULE.format(k=max_candidates_per_track),
    }


def confirmation_pipeline_ids(
    screening: pd.DataFrame,
    *,
    max_candidates_per_track: int = 5,
) -> tuple[str, ...]:
    """Выбрать якоря и до ``k`` кандидатов каждого контура (см. правило)."""

    finalists = select_confirmation_finalists(
        screening, max_candidates_per_track=max_candidates_per_track
    )
    return tuple(sorted(set(finalists["pipeline"]))) if not finalists.empty else ()


def select_rnd_leaders(aggregates: pd.DataFrame) -> pd.DataFrame:
    """Ранжировать только внутри одного estimand, эффекта и прошедшего экрана."""

    if aggregates.empty or "inference_quality" not in aggregates:
        return pd.DataFrame()
    eligible = aggregates.loc[
        (aggregates["inference_quality"] == "good")
        & aggregates["scientific_sample_size_ok"].fillna(False).astype(bool)
        & aggregates["h1_rmse"].notna()
    ].copy()
    if eligible.empty:
        return eligible
    group = [
        "world_id",
        "information_mode",
        "temporal_access",
        "ranking_estimand_id",
        "effect_size",
    ]
    eligible["rank_rmse"] = eligible.groupby(group)["h1_rmse"].rank(method="min")
    eligible["rank_power"] = eligible.groupby(group)["h1_rejection_rate"].rank(
        method="min", ascending=False
    )
    eligible["pareto_score"] = eligible["rank_rmse"] + eligible["rank_power"]
    return eligible.sort_values([*group, "pareto_score", "runtime_median_ms"])

