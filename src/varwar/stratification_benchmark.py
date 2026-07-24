from __future__ import annotations

import math
import time
import tracemalloc
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec, WrappedEstimator
from varwar.estimators import (
    DifferenceInMeansEstimator,
    HypExABTestEstimator,
    HypExCupacEstimator,
    StratifiedDifferenceInMeansEstimator,
)
from varwar.feature_sets import CupacFeaturePolicy, default_cupac_feature_sets
from varwar.stratification import (
    ABProportionalVarianceObjective,
    AtomicStrataSummary,
    GeneticStrataReducer,
    aggregate_atomic_strata,
)
from varwar.transforms import CupedTransformer, ZonedCupacTransformer


def _cupac_variants(spec: ExperimentSpec) -> list[CupacFeaturePolicy]:
    return default_cupac_feature_sets(spec)


def _apply_mapping(
    data: pd.DataFrame,
    summary: AtomicStrataSummary,
    labels: np.ndarray,
    output_col: str,
) -> pd.DataFrame:
    mapping = summary.keys.copy()
    mapping[output_col] = np.asarray(labels, dtype=np.int32)
    position = "__varwar_position"
    while position in data.columns:
        position += "_"
    left = data.loc[:, summary.category_cols].copy()
    left[position] = np.arange(len(data), dtype=np.int64)
    merged = left.merge(
        mapping,
        how="left",
        on=list(summary.category_cols),
        sort=False,
        validate="many_to_one",
    ).sort_values(position, kind="stable")
    raw = merged[output_col].to_numpy(dtype=float, copy=True)
    unknown = ~np.isfinite(raw)
    if unknown.any():
        macro = aggregate_atomic_strata(summary, labels)
        largest = int(macro.loc[macro["count"].idxmax(), "stratum"])
        raw[unknown] = largest
    result = data.copy(deep=False)
    result[output_col] = pd.Categorical(raw.astype(np.int32))
    return result


def _baseline_partitions(
    summary: AtomicStrataSummary,
    n_strata: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    n_groups = min(n_strata, summary.n_atomic_strata)
    atomic = np.arange(summary.n_atomic_strata, dtype=np.int32)

    order = np.argsort(summary.means, kind="stable")
    mean_sorted = np.empty(summary.n_atomic_strata, dtype=np.int32)
    for group, atoms in enumerate(np.array_split(order, n_groups)):
        mean_sorted[atoms] = group

    rng = np.random.default_rng(random_state)
    random_order = rng.permutation(summary.n_atomic_strata)
    random_labels = np.empty(summary.n_atomic_strata, dtype=np.int32)
    random_labels[random_order] = np.arange(summary.n_atomic_strata) % n_groups
    return {
        "atomic_strata": atomic,
        "random_reduction": random_labels,
        "mean_sorted_reduction": mean_sorted,
    }


def _mapping_validation_score(
    validation_data: pd.DataFrame | None,
    summary: AtomicStrataSummary,
    labels: np.ndarray,
    target_col: str,
    output_col: str,
    objective: Any,
) -> tuple[float, str | None]:
    if validation_data is None:
        return math.nan, None
    try:
        prepared = _apply_mapping(validation_data, summary, labels, output_col)
        target = np.asarray(
            pd.to_numeric(prepared[target_col], errors="coerce"),
            dtype=float,
        )
        strata = np.asarray(prepared[output_col].astype(int), dtype=np.int32)
        valid = np.isfinite(target)
        grouped = (
            pd.DataFrame({"stratum": strata[valid], "target": target[valid]})
            .groupby("stratum", sort=True)["target"]
            .agg(["count", "var"])
        )
        expected_groups = int(np.unique(labels).size)
        if len(grouped) != expected_groups or (grouped["count"] < 2).any():
            return math.nan, "validation data has fewer than two rows in a macro stratum"
        evaluation = objective.evaluate(
            np.asarray(grouped["count"], dtype=np.int64),
            np.asarray(grouped["var"].fillna(0.0), dtype=float),
        )
        return float(evaluation.score), None
    except (KeyError, TypeError, ValueError) as exc:
        return math.nan, str(exc)


def _estimate_row(
    pipeline: str,
    preprocessor: str,
    objective: str,
    estimator: Any,
    data: pd.DataFrame,
    spec: ExperimentSpec,
    reference_data: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        if isinstance(estimator, WrappedEstimator):
            result = estimator.estimate(
                data,
                spec,
                auxiliary_data=reference_data,
            )
        elif callable(getattr(estimator, "estimate_with_auxiliary", None)):
            result = estimator.estimate_with_auxiliary(data, spec, reference_data)
        else:
            result = estimator.estimate(data, spec)
        row = result.to_record()
        row["status"] = "ok"
    except Exception as exc:
        row = {
            "estimator": getattr(estimator, "name", estimator.__class__.__name__),
            "status": "failed",
            "error": str(exc),
        }
    row.update(metadata)
    row.update(
        {
            "pipeline": pipeline,
            "preprocessor": preprocessor,
            "objective": objective,
            "estimator_runtime_ms": (time.perf_counter() - start) * 1000.0,
        }
    )
    return row


def run_stratification_race(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    spec: ExperimentSpec,
    category_cols: tuple[str, ...],
    reference_target_col: str,
    n_strata: int,
    reducers: Mapping[str, GeneticStrataReducer] | None = None,
    include_target_transformers: bool = True,
    measure_memory: bool = False,
    random_state: int = 42,
    validation_data: pd.DataFrame | None = None,
    true_effect: float | None = None,
) -> pd.DataFrame:
    """Сравнить design preprocessors отдельно от обычной гонки эстиматоров.

    Reducer обучается на ``reference_data`` только по
    ``reference_target_col``. В одно-датасетной схеме сюда передаётся тот же
    frame, что и в ``data``, а ``reference_target_col`` указывает на pre-target.
    Текущий экспериментальный target не используется при назначении страт.
    """

    if n_strata < 1:
        raise ValueError("n_strata must be positive.")
    output_col = "__varwar_race_stratum"
    race_spec = spec.with_stratum(output_col)
    rows: list[dict[str, Any]] = []

    baseline_estimators: list[Any] = [
        DifferenceInMeansEstimator(),
        HypExABTestEstimator(),
    ]
    if include_target_transformers:
        baseline_estimators.extend(
            [
                WrappedEstimator(
                    CupedTransformer(fit_data="experiment_control"),
                    HypExABTestEstimator(),
                    name="cuped_control__hypex_ab",
                ),
                HypExCupacEstimator(),
            ]
        )
        for policy in _cupac_variants(spec):
            baseline_estimators.append(
                WrappedEstimator(
                    ZonedCupacTransformer(policy=policy),
                    HypExABTestEstimator(),
                    name=f"{policy.method_id}__hypex_ab",
                )
            )
    for estimator in baseline_estimators:
        rows.append(
            _estimate_row(
                estimator.name,
                "none",
                "none",
                estimator,
                data,
                spec,
                reference_data,
                {
                    "preprocess_runtime_ms": 0.0,
                    "preprocess_peak_memory_bytes": 0,
                    "train_score": math.nan,
                    "continuous_score": math.nan,
                    "validation_score": math.nan,
                    "n_atomic_strata": math.nan,
                    "n_strata": 1,
                },
            )
        )

    start = time.perf_counter()
    summary = AtomicStrataSummary.from_frame(
        reference_data,
        category_cols,
        reference_target_col,
    )
    summary_runtime = (time.perf_counter() - start) * 1000.0
    ab_objective = ABProportionalVarianceObjective()
    for name, labels in _baseline_partitions(summary, n_strata, random_state).items():
        macro = aggregate_atomic_strata(summary, labels)
        evaluation = ab_objective.evaluate(
            np.asarray(macro["count"], dtype=np.int64),
            np.asarray(macro["variance"], dtype=float),
        )
        validation_score, validation_error = _mapping_validation_score(
            validation_data,
            summary,
            labels,
            reference_target_col,
            output_col,
            ab_objective,
        )
        try:
            prepared = _apply_mapping(data, summary, labels, output_col)
            row = _estimate_row(
                f"{name}__stratified_mean_diff",
                name,
                "ab_proportional",
                StratifiedDifferenceInMeansEstimator(),
                prepared,
                race_spec,
                reference_data,
                {
                    "preprocess_runtime_ms": summary_runtime,
                    "preprocess_peak_memory_bytes": 0,
                    "train_score": evaluation.score,
                    "continuous_score": evaluation.continuous_score,
                    "validation_score": validation_score,
                    "validation_error": validation_error,
                    "n_atomic_strata": summary.n_atomic_strata,
                    "n_strata": len(macro),
                },
            )
        except Exception as exc:
            row = {
                "pipeline": f"{name}__stratified_mean_diff",
                "preprocessor": name,
                "objective": "ab_proportional",
                "estimator": "stratified_mean_diff",
                "status": "failed",
                "error": str(exc),
                "preprocess_runtime_ms": summary_runtime,
                "preprocess_peak_memory_bytes": 0,
                "train_score": evaluation.score,
                "continuous_score": evaluation.continuous_score,
                "validation_score": validation_score,
                "validation_error": validation_error,
                "n_atomic_strata": summary.n_atomic_strata,
                "n_strata": len(macro),
                "estimator_runtime_ms": 0.0,
            }
        rows.append(row)

    if reducers is None:
        reducers = {
            "genetic_strata_reducer": GeneticStrataReducer(
                category_cols=category_cols,
                reference_target_col=reference_target_col,
                n_strata=n_strata,
                objective=ABProportionalVarianceObjective(),
                population_size=32,
                max_generations=80,
                elite_size=4,
                patience=15,
                random_state=random_state,
                unknown_policy="largest",
                output_col=output_col,
            ),
        }

    for name, reducer in reducers.items():
        peak_memory = 0
        if measure_memory:
            tracemalloc.start()
        start = time.perf_counter()
        try:
            reducer.fit(reference_data, validation_data=validation_data)
            prepared = reducer.augment(data)
            fit_error = None
        except Exception as exc:
            prepared = None
            fit_error = str(exc)
        fit_runtime = (time.perf_counter() - start) * 1000.0
        if measure_memory:
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        if fit_error is not None or reducer.result_ is None or prepared is None:
            rows.append(
                {
                    "pipeline": f"{name}__stratified_mean_diff",
                    "preprocessor": name,
                    "objective": reducer.objective.name,
                    "estimator": "stratified_mean_diff",
                    "status": "failed",
                    "error": fit_error or "Reducer did not produce a result.",
                    "preprocess_runtime_ms": fit_runtime,
                    "preprocess_peak_memory_bytes": peak_memory,
                    "estimator_runtime_ms": 0.0,
                }
            )
            continue
        metadata = {
            "preprocess_runtime_ms": fit_runtime,
            "preprocess_peak_memory_bytes": peak_memory,
            "train_score": reducer.result_.train_score,
            "continuous_score": reducer.result_.continuous_score,
            "validation_score": reducer.result_.validation_score,
            "n_atomic_strata": reducer.result_.n_atomic_strata,
            "n_strata": reducer.result_.n_strata,
            "generations_run": reducer.result_.generations_run,
            "converged": reducer.result_.converged,
            "analysis_rows": len(prepared),
        }
        stratum_weights = None
        total_reference = float(reducer.macro_summary_["count"].sum())
        stratum_weights = {
            int(row.stratum): float(row.count / total_reference)
            for row in reducer.macro_summary_.itertuples(index=False)
        }

        def stratified_estimator() -> StratifiedDifferenceInMeansEstimator:
            return StratifiedDifferenceInMeansEstimator(stratum_weights=stratum_weights)

        estimators: list[Any] = [stratified_estimator()]
        if include_target_transformers:
            estimators.append(
                WrappedEstimator(
                    CupedTransformer(fit_data="experiment_control"),
                    stratified_estimator(),
                    name="cuped_control__stratified",
                )
            )
            for policy in _cupac_variants(spec):
                estimators.append(
                    WrappedEstimator(
                        ZonedCupacTransformer(policy=policy),
                        stratified_estimator(),
                        name=f"{policy.method_id}__stratified",
                    )
                )
        for estimator in estimators:
            if name == "genetic_strata_reducer":
                canonical_pipeline = {
                    "stratified_mean_diff": "ga_strata__stratified",
                    "cuped_control__stratified": (
                        "ga_strata__cuped_control__stratified"
                    ),
                    "cupac_linear_A__stratified": (
                        "ga_strata__cupac_linear_A__stratified"
                    ),
                    "cupac_linear_A_plus_B__stratified": (
                        "ga_strata__cupac_linear_A_plus_B__stratified"
                    ),
                    "cupac_linear_A_plus_B_plus_C__stratified": (
                        "ga_strata__cupac_linear_A_plus_B_plus_C__stratified"
                    ),
                }.get(estimator.name, f"ga_strata__{estimator.name}")
            else:
                canonical_pipeline = f"{name}__{estimator.name}"
            rows.append(
                _estimate_row(
                    canonical_pipeline,
                    name,
                    reducer.objective.name,
                    estimator,
                    prepared,
                    race_spec,
                    reference_data,
                    metadata,
                )
            )
    result = pd.DataFrame(rows)
    if true_effect is not None:
        effect = pd.to_numeric(result.get("effect"), errors="coerce")
        result["true_effect"] = float(true_effect)
        result["bias"] = effect - float(true_effect)
    return result


def summarize_stratification_trials(
    trials: pd.DataFrame,
    true_effect: float,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Свести повторные R&D-прогоны в bias, RMSE, FPR или power.

    Ожидается результат нескольких вызовов ``run_stratification_race`` с
    добавленной caller-ом колонкой ``trial``. Неуспешные строки исключаются,
    но их доля сохраняется в ``success_rate``.
    """

    required = {
        "pipeline",
        "preprocessor",
        "objective",
        "estimator",
        "status",
        "effect",
        "standard_error",
        "p_value",
    }
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"Missing trial columns: {sorted(missing)}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one.")

    groups = ["pipeline", "preprocessor", "objective", "estimator"]
    rows: list[dict[str, Any]] = []
    for keys, frame in trials.groupby(groups, dropna=False, sort=False):
        ok = frame.loc[frame["status"] == "ok"].copy()
        effect = np.asarray(pd.to_numeric(ok["effect"], errors="coerce"), dtype=float)
        se = np.asarray(
            pd.to_numeric(ok["standard_error"], errors="coerce"),
            dtype=float,
        )
        p_value = np.asarray(pd.to_numeric(ok["p_value"], errors="coerce"), dtype=float)
        finite = np.isfinite(effect) & np.isfinite(se) & np.isfinite(p_value)
        effect = effect[finite]
        se = se[finite]
        p_value = p_value[finite]
        n_total = int(len(frame))
        n_success = int(effect.size)
        row = dict(zip(groups, keys))
        row.update(
            {
                "n_trials": n_total,
                "n_success": n_success,
                "success_rate": n_success / n_total if n_total else math.nan,
                "bias": float(effect.mean() - true_effect) if n_success else math.nan,
                "rmse": (
                    float(np.sqrt(np.mean((effect - true_effect) ** 2)))
                    if n_success
                    else math.nan
                ),
                "mean_standard_error": float(se.mean()) if n_success else math.nan,
                "empirical_standard_deviation": (
                    float(effect.std(ddof=1)) if n_success > 1 else math.nan
                ),
                "rejection_rate": float(np.mean(p_value < alpha)) if n_success else math.nan,
                "rate_kind": "false_positive_rate" if true_effect == 0.0 else "power",
                "true_effect": float(true_effect),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
