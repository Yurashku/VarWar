from __future__ import annotations

import gc
import math
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from varwar.benchmark import run_estimator_race
from varwar.core import ExperimentSpec, WrappedEstimator
from varwar.estimators import (
    DifferenceInMeansEstimator,
    HypExABTestEstimator,
    HypExCupacEstimator,
    MLRateEstimator,
    RegressionAdjustmentEstimator,
    StratifiedDifferenceInMeansEstimator,
    TrimmedMeanEstimator,
    WinsorizedMeanEstimator,
)
from varwar.stratification import (
    ABProportionalVarianceObjective,
    GeneticStrataReducer,
)
from varwar.synthetic import (
    SyntheticConfig,
    generate_historical_controls,
    generate_synthetic_ab,
)
from varwar.transforms import CupedTransformer, LinearCupacTransformer


@dataclass(frozen=True)
class SyntheticPipelineRaceConfig:
    """Configuration of a paired H0/H1 race over complete analysis pipelines."""

    n_trials: int = 100
    n: int = 2_000
    alternative_effect: float = 0.15
    covariate_signal: float = 0.75
    noise_scale: float = 1.0
    heavy_tail: bool = False
    seed: int = 20_260_715
    category_cols: tuple[str, ...] = ("region", "device", "activity_band")
    n_strata: int = 6
    population_size: int = 16
    max_generations: int = 25
    elite_size: int = 3
    patience: int = 7
    min_macro_stratum_size: int = 8
    mutation_probability: float = 0.2
    prefer_pypi_hypex: bool = True
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive.")
        if self.n < 8:
            raise ValueError("n must be at least eight.")
        if not 0 < self.n_strata <= self.n:
            raise ValueError("n_strata must be in [1, n].")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between zero and one.")


_FULL_PIPELINES = (
    "welch_mean_diff",
    "hypex_ab",
    "cuped+hypex_ab",
    "hypex_cupac",
    "sklearn_cupac_A+hypex_ab",
    "sklearn_cupac_A_plus_B+hypex_ab",
    "sklearn_cupac_A_plus_B_plus_C+hypex_ab",
    "regression_adjustment",
    "ml_rate",
    "trimmed_mean_diff",
    "winsorized_mean_diff",
)


def _spec(config: SyntheticPipelineRaceConfig) -> ExperimentSpec:
    return ExperimentSpec(
        unit_col="unit_id",
        pre_target_col="y_pre",
        covariate_cols=("y_pre", "x1", "x2"),
        pre_treatment_cols=("y_pre", "x1", "x2"),
        expert_safe_in_time_cols=("calendar_index", "market_index"),
        gated_in_time_cols=("operational_load",),
        alpha=config.alpha,
    )


def _hypex(config: SyntheticPipelineRaceConfig) -> HypExABTestEstimator:
    return HypExABTestEstimator(prefer_pypi=config.prefer_pypi_hypex)


def _full_estimators(
    config: SyntheticPipelineRaceConfig,
    spec: ExperimentSpec,
) -> list[Any]:
    a = spec.pre_treatment_features()
    b = tuple(dict.fromkeys(a + spec.expert_safe_in_time_cols))
    c = tuple(dict.fromkeys(b + spec.gated_in_time_cols))
    return [
        DifferenceInMeansEstimator(),
        _hypex(config),
        WrappedEstimator(CupedTransformer(), _hypex(config)),
        HypExCupacEstimator(
            cupac_features=a,
            prefer_pypi=config.prefer_pypi_hypex,
        ),
        WrappedEstimator(
            LinearCupacTransformer(feature_cols=a, name="sklearn_cupac_A"),
            _hypex(config),
        ),
        WrappedEstimator(
            LinearCupacTransformer(feature_cols=b, name="sklearn_cupac_A_plus_B"),
            _hypex(config),
        ),
        WrappedEstimator(
            LinearCupacTransformer(
                feature_cols=c,
                name="sklearn_cupac_A_plus_B_plus_C",
            ),
            _hypex(config),
        ),
        RegressionAdjustmentEstimator(),
        MLRateEstimator(),
        TrimmedMeanEstimator(),
        WinsorizedMeanEstimator(),
    ]


def _stratified_estimators(
    config: SyntheticPipelineRaceConfig,
    spec: ExperimentSpec,
    weights: dict[Any, float],
) -> list[Any]:
    def base() -> StratifiedDifferenceInMeansEstimator:
        return StratifiedDifferenceInMeansEstimator(stratum_weights=weights)

    a = spec.pre_treatment_features()
    c = tuple(
        dict.fromkeys(
            a + spec.expert_safe_in_time_cols + spec.gated_in_time_cols
        )
    )
    return [
        base(),
        WrappedEstimator(CupedTransformer(), base()),
        WrappedEstimator(
            LinearCupacTransformer(feature_cols=a, name="sklearn_cupac_A"),
            base(),
        ),
        WrappedEstimator(
            LinearCupacTransformer(
                feature_cols=c,
                name="sklearn_cupac_A_plus_B_plus_C",
            ),
            base(),
        ),
    ]


def _fit_reducer(
    reference: pd.DataFrame,
    config: SyntheticPipelineRaceConfig,
    objective: Any,
    random_state: int,
    output_col: str,
    measure_memory: bool,
) -> tuple[GeneticStrataReducer, float, int | None]:
    reducer = GeneticStrataReducer(
        category_cols=config.category_cols,
        reference_target_col="y",
        n_strata=config.n_strata,
        objective=objective,
        min_stratum_size=config.min_macro_stratum_size,
        population_size=config.population_size,
        max_generations=config.max_generations,
        elite_size=config.elite_size,
        mutation_probability=config.mutation_probability,
        patience=config.patience,
        random_state=random_state,
        unknown_policy="largest",
        output_col=output_col,
    )
    start = time.perf_counter()
    if measure_memory:
        gc.collect()
        tracemalloc.start()
    try:
        reducer.fit(reference)
    finally:
        if measure_memory:
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            peak_memory_bytes = None
    runtime_ms = (time.perf_counter() - start) * 1_000.0
    return (
        reducer,
        runtime_ms,
        int(peak_memory_bytes) if peak_memory_bytes is not None else None,
    )


def _population_weights(data: pd.DataFrame, stratum_col: str) -> dict[Any, float]:
    counts = data[stratum_col].value_counts(sort=False, dropna=False)
    total = float(counts.sum())
    return {key: float(value / total) for key, value in counts.items()}


def _contextualize(
    frame: pd.DataFrame,
    *,
    pipeline_prefix: str,
    data_scenario: str,
    track: str,
    hypothesis: str,
    true_effect: float,
    trial: int,
    seed: int,
    preprocessing_runtime_ms: float = 0.0,
    preprocessing_peak_memory_bytes: int | None = None,
    preprocessor: str = "none",
) -> pd.DataFrame:
    result = frame.copy()
    result["pipeline"] = pipeline_prefix + result["estimator"].astype(str)
    result["data_scenario"] = data_scenario
    result["track"] = track
    result["hypothesis"] = hypothesis
    result["true_effect"] = float(true_effect)
    result["trial"] = int(trial)
    result["seed"] = int(seed)
    result["preprocessor"] = preprocessor
    result["preprocessing_runtime_ms"] = float(preprocessing_runtime_ms)
    result["end_to_end_runtime_ms"] = (
        pd.to_numeric(result["runtime_ms"], errors="coerce")
        + float(preprocessing_runtime_ms)
    )
    estimator_peak = pd.to_numeric(
        result.get(
            "peak_memory_bytes",
            pd.Series(float("nan"), index=result.index),
        ),
        errors="coerce",
    )
    result["preprocessing_peak_memory_bytes"] = preprocessing_peak_memory_bytes
    if preprocessing_peak_memory_bytes is None:
        result["end_to_end_peak_memory_bytes"] = estimator_peak
    else:
        result["end_to_end_peak_memory_bytes"] = np.maximum(
            estimator_peak.fillna(0.0).to_numpy(dtype=float),
            float(preprocessing_peak_memory_bytes),
        )
    n_control = pd.to_numeric(result.get("n_control"), errors="coerce")
    n_treatment = pd.to_numeric(result.get("n_treatment"), errors="coerce")
    result["analysis_rows"] = n_control + n_treatment
    return result


def _evaluate(
    data: pd.DataFrame,
    spec: ExperimentSpec,
    estimators: list[Any],
    reference: pd.DataFrame,
    measure_memory: bool,
    **context: Any,
) -> pd.DataFrame:
    frame = run_estimator_race(
        data,
        spec,
        estimators=estimators,
        auxiliary_data=reference,
        measure_memory=measure_memory,
    )
    return _contextualize(frame, **context)


def _experiment_pair(
    config: SyntheticPipelineRaceConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_config = SyntheticConfig(
        n=config.n,
        effect=0.0,
        covariate_signal=config.covariate_signal,
        noise_scale=config.noise_scale,
        seed=seed,
        heavy_tail=config.heavy_tail,
    )
    null = generate_synthetic_ab(base_config)
    alternative = null.copy()
    alternative["y"] = (
        np.asarray(alternative["y0"], dtype=float)
        + np.asarray(alternative["treatment"], dtype=float)
        * config.alternative_effect
    )
    reference = generate_historical_controls(base_config)
    return null, alternative, reference


def run_synthetic_pipeline_race(
    config: SyntheticPipelineRaceConfig = SyntheticPipelineRaceConfig(),
    progress: Callable[[int, int], None] | None = None,
    *,
    measure_memory: bool = False,
) -> pd.DataFrame:
    """Run paired null/alternative trials over comparable final pipelines.

    Every pipeline uses all experiment rows. Every target model and strata
    mapping is fit on an independent historical sample.
    """

    spec = _spec(config)
    rows: list[pd.DataFrame] = []
    data_scenario = "linear_heavy_tail" if config.heavy_tail else "linear_gaussian"
    total = config.n_trials

    for trial in range(config.n_trials):
        seed = config.seed + trial
        null, alternative, reference = _experiment_pair(config, seed)
        ab_reducer, ab_runtime, ab_peak_memory = _fit_reducer(
            reference,
            config,
            ABProportionalVarianceObjective(),
            seed + 100_000,
            "__varwar_ab_stratum",
            measure_memory,
        )
        for hypothesis, true_effect, experiment in (
            ("H0", 0.0, null),
            ("H1", config.alternative_effect, alternative),
        ):
            common = {
                "data_scenario": data_scenario,
                "hypothesis": hypothesis,
                "true_effect": true_effect,
                "trial": trial,
                "seed": seed,
            }
            full = _evaluate(
                experiment,
                spec,
                _full_estimators(config, spec),
                reference,
                measure_memory=measure_memory,
                pipeline_prefix="",
                track="full_traffic",
                **common,
            )
            rows.append(full.loc[full["pipeline"].isin(_FULL_PIPELINES)])

            prepared_ab = ab_reducer.augment(experiment)
            ab_weights = _population_weights(prepared_ab, ab_reducer.output_col)
            ga_ab = _evaluate(
                prepared_ab,
                spec.with_stratum(ab_reducer.output_col),
                _stratified_estimators(config, spec, ab_weights),
                reference,
                measure_memory=measure_memory,
                pipeline_prefix="genetic_strata_ab+",
                track="full_traffic",
                preprocessing_runtime_ms=ab_runtime,
                preprocessing_peak_memory_bytes=ab_peak_memory,
                preprocessor="genetic_strata_ab",
                **common,
            )
            rows.append(ga_ab)

        if progress is not None:
            progress(trial + 1, total)

    return pd.concat(rows, ignore_index=True, sort=False)


def profile_synthetic_pipeline_memory(
    config: SyntheticPipelineRaceConfig = SyntheticPipelineRaceConfig(),
) -> pd.DataFrame:
    """Profile Python allocation peaks in one reproducible paired trial.

    Memory instrumentation is intentionally separated from the Monte Carlo
    race so tracemalloc overhead cannot distort the reported production-like
    runtime. An unmeasured warm-up removes one-time import allocations and
    makes the profile insensitive to pipeline order. Native NumPy/BLAS buffers
    are outside this metric.
    """

    profile_config = replace(config, n_trials=1)
    run_synthetic_pipeline_race(profile_config, measure_memory=False)
    profile = run_synthetic_pipeline_race(
        profile_config,
        measure_memory=True,
    )
    ok = profile.loc[profile["status"] == "ok"].copy()
    keys = ["data_scenario", "track", "pipeline"]
    result = (
        ok.groupby(keys, sort=False, dropna=False)[
            "end_to_end_peak_memory_bytes"
        ]
        .max()
        .rename("peak_python_memory_bytes")
        .reset_index()
    )
    result["peak_python_memory_mib"] = (
        result["peak_python_memory_bytes"] / (1024.0**2)
    )
    result["memory_profile_trials"] = 1
    result["memory_profile_seed"] = config.seed
    return result


def _hypothesis_metrics(
    frame: pd.DataFrame,
    true_effect: float,
    alpha: float,
) -> dict[str, Any]:
    ok = frame.loc[frame["status"] == "ok"].copy()
    effect = np.asarray(pd.to_numeric(ok.get("effect"), errors="coerce"), dtype=float)
    se = np.asarray(
        pd.to_numeric(ok.get("standard_error"), errors="coerce"),
        dtype=float,
    )
    p_value = np.asarray(
        pd.to_numeric(ok.get("p_value"), errors="coerce"),
        dtype=float,
    )
    finite = np.isfinite(effect) & np.isfinite(se) & np.isfinite(p_value)
    ok = ok.loc[finite]
    effect = effect[finite]
    se = se[finite]
    p_value = p_value[finite]
    ci_low = np.asarray(pd.to_numeric(ok.get("ci_low"), errors="coerce"), dtype=float)
    ci_high = np.asarray(pd.to_numeric(ok.get("ci_high"), errors="coerce"), dtype=float)
    coverage = np.isfinite(ci_low) & np.isfinite(ci_high)
    covered = coverage & (ci_low <= true_effect) & (true_effect <= ci_high)
    backends = sorted(
        {
            str(value)
            for value in ok.get("diag_backend", pd.Series(dtype=object)).dropna()
        }
    )
    runtime = np.asarray(
        pd.to_numeric(ok.get("end_to_end_runtime_ms"), errors="coerce"),
        dtype=float,
    )
    analysis_rows = np.asarray(
        pd.to_numeric(ok.get("analysis_rows"), errors="coerce"),
        dtype=float,
    )
    return {
        "n_total": int(len(frame)),
        "n_success": int(effect.size),
        "success_rate": float(effect.size / len(frame)) if len(frame) else math.nan,
        "mean_effect": float(effect.mean()) if effect.size else math.nan,
        "bias": float(effect.mean() - true_effect) if effect.size else math.nan,
        "empirical_sd": float(effect.std(ddof=1)) if effect.size > 1 else math.nan,
        "mean_se": float(se.mean()) if se.size else math.nan,
        "rejection_rate": float(np.mean(p_value < alpha)) if p_value.size else math.nan,
        "coverage": float(covered.sum() / coverage.sum()) if coverage.any() else math.nan,
        "median_runtime_ms": (
            float(np.nanmedian(runtime)) if np.isfinite(runtime).any() else math.nan
        ),
        "median_analysis_rows": (
            float(np.nanmedian(analysis_rows))
            if np.isfinite(analysis_rows).any()
            else math.nan
        ),
        "backend": ",".join(backends) if backends else "native",
    }


def _binomial_calibration_pvalue(successes: int, trials: int, alpha: float) -> float:
    if trials < 1:
        return math.nan
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(scipy_stats.binomtest(successes, trials, alpha).pvalue)
    except Exception:
        variance = trials * alpha * (1.0 - alpha)
        if variance <= 0.0:
            return math.nan
        z_value = abs(successes - trials * alpha) / math.sqrt(variance)
        return math.erfc(z_value / math.sqrt(2.0))


def _binomial_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1:
        return math.nan, math.nan
    try:
        from scipy import stats as scipy_stats  # type: ignore

        interval = scipy_stats.binomtest(successes, trials).proportion_ci(
            confidence_level=0.95,
            method="exact",
        )
        return float(interval.low), float(interval.high)
    except Exception:
        proportion = successes / trials
        z_value = 1.959963984540054
        denominator = 1.0 + z_value**2 / trials
        center = (proportion + z_value**2 / (2.0 * trials)) / denominator
        half_width = z_value * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_value**2 / (4.0 * trials**2)
        ) / denominator
        return max(0.0, center - half_width), min(1.0, center + half_width)


def summarize_synthetic_pipeline_race(
    trials: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aggregate paired H0/H1 trials and compute baseline-relative efficiency."""

    required = {
        "data_scenario",
        "track",
        "pipeline",
        "hypothesis",
        "true_effect",
        "status",
    }
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"Missing pipeline-race columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    groups = ["data_scenario", "track", "pipeline"]
    for keys, frame in trials.groupby(groups, sort=False, dropna=False):
        null = frame.loc[frame["hypothesis"] == "H0"]
        alternative = frame.loc[frame["hypothesis"] == "H1"]
        if null.empty or alternative.empty:
            continue
        h0 = _hypothesis_metrics(null, 0.0, alpha)
        true_effect = float(alternative["true_effect"].iloc[0])
        h1 = _hypothesis_metrics(alternative, true_effect, alpha)
        rejected = int(round(h0["rejection_rate"] * h0["n_success"]))
        calibration_p = _binomial_calibration_pvalue(
            rejected,
            h0["n_success"],
            alpha,
        )
        fpr_low, fpr_high = _binomial_interval(rejected, h0["n_success"])
        power_rejected = int(round(h1["rejection_rate"] * h1["n_success"]))
        power_low, power_high = _binomial_interval(
            power_rejected,
            h1["n_success"],
        )
        rows.append(
            {
                "data_scenario": keys[0],
                "track": keys[1],
                "pipeline": keys[2],
                "n_trials": min(h0["n_total"], h1["n_total"]),
                "success_rate": min(h0["success_rate"], h1["success_rate"]),
                "analysis_rows": h1["median_analysis_rows"],
                "true_effect": true_effect,
                "mean_effect_h1": h1["mean_effect"],
                "bias_h0": h0["bias"],
                "bias_h1": h1["bias"],
                "empirical_sd_h1": h1["empirical_sd"],
                "mean_se_h1": h1["mean_se"],
                "se_to_empirical_sd": (
                    h1["mean_se"] / h1["empirical_sd"]
                    if h1["empirical_sd"] > 0.0
                    else math.nan
                ),
                "false_positive_rate": h0["rejection_rate"],
                "fpr_ci_low": fpr_low,
                "fpr_ci_high": fpr_high,
                "fpr_calibration_p_value": calibration_p,
                "fpr_calibrated": bool(calibration_p >= alpha),
                "power": h1["rejection_rate"],
                "power_ci_low": power_low,
                "power_ci_high": power_high,
                "coverage_h1": h1["coverage"],
                "median_runtime_ms": float(
                    np.nanmedian([h0["median_runtime_ms"], h1["median_runtime_ms"]])
                ),
                "backend": h1["backend"],
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["variance_reduction_vs_baseline"] = math.nan
    baselines = {"full_traffic": "welch_mean_diff"}
    for (scenario, track), index in summary.groupby(
        ["data_scenario", "track"], sort=False
    ).groups.items():
        baseline_id = baselines.get(str(track))
        block = summary.loc[index]
        baseline = block.loc[block["pipeline"] == baseline_id, "empirical_sd_h1"]
        if baseline.empty or not math.isfinite(float(baseline.iloc[0])):
            continue
        baseline_variance = float(baseline.iloc[0]) ** 2
        if baseline_variance <= 0.0:
            continue
        summary.loc[index, "variance_reduction_vs_baseline"] = 1.0 - (
            summary.loc[index, "empirical_sd_h1"] ** 2 / baseline_variance
        )
    return summary.sort_values(
        ["data_scenario", "track", "fpr_calibrated", "empirical_sd_h1"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


_PIPELINE_LABELS_RU = {
    "welch_mean_diff": "Welch t-test",
    "hypex_ab": "HypEx A/B",
    "cuped+hypex_ab": "CUPED → HypEx A/B",
    "hypex_cupac": "HypEx CUPAC",
    "sklearn_cupac_A+hypex_ab": "CUPAC A → HypEx A/B",
    "sklearn_cupac_A_plus_B+hypex_ab": "CUPAC A+B → HypEx A/B",
    "sklearn_cupac_A_plus_B_plus_C+hypex_ab": "CUPAC A+B+C → HypEx A/B",
    "regression_adjustment": "Regression adjustment",
    "ml_rate": "ML-RATE",
    "trimmed_mean_diff": "Trimmed mean",
    "winsorized_mean_diff": "Winsorized mean",
    "genetic_strata_ab+stratified_mean_diff": "GA(AB) → stratified",
    "genetic_strata_ab+cuped+stratified_mean_diff": "GA(AB) → CUPED → stratified",
    "genetic_strata_ab+sklearn_cupac_A+stratified_mean_diff": "GA(AB) → CUPAC A → stratified",
    "genetic_strata_ab+sklearn_cupac_A_plus_B_plus_C+stratified_mean_diff": "GA(AB) → CUPAC A+B+C → stratified",
}


def format_pipeline_summary_ru(summary: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation table with explicit Russian column names."""

    result = pd.DataFrame(
        {
            "Сценарий": summary["data_scenario"].map(
                {
                    "linear_gaussian": "Линейный, нормальный шум",
                    "linear_heavy_tail": "Линейный, тяжёлые хвосты",
                }
            ).fillna(summary["data_scenario"]),
            "Контур": summary["track"].map(
                {
                    "full_traffic": "Полный трафик",
                }
            ).fillna(summary["track"]),
            "Пайплайн": summary["pipeline"].map(_PIPELINE_LABELS_RU).fillna(
                summary["pipeline"]
            ),
            "N анализа": summary["analysis_rows"].round().astype("Int64"),
            "Успешно, %": (100.0 * summary["success_rate"]).round(1),
            "Оценка при H1": summary["mean_effect_h1"].round(4),
            "Смещение при H1": summary["bias_h1"].round(4),
            "Эмпирическое SD": summary["empirical_sd_h1"].round(4),
            "Средний SE": summary["mean_se_h1"].round(4),
            "SE / эмп. SD": summary["se_to_empirical_sd"].round(3),
            "Снижение дисперсии, %": (
                100.0 * summary["variance_reduction_vs_baseline"]
            ).round(1),
            "FPR, % [95% ДИ]": [
                f"{100.0 * value:.1f} [{100.0 * low:.1f}; {100.0 * high:.1f}]"
                for value, low, high in zip(
                    summary["false_positive_rate"],
                    summary["fpr_ci_low"],
                    summary["fpr_ci_high"],
                )
            ],
            "FPR калиброван": summary["fpr_calibrated"].map(
                {True: "да", False: "нет"}
            ),
            "Мощность, % [95% ДИ]": [
                f"{100.0 * value:.1f} [{100.0 * low:.1f}; {100.0 * high:.1f}]"
                for value, low, high in zip(
                    summary["power"],
                    summary["power_ci_low"],
                    summary["power_ci_high"],
                )
            ],
            "Покрытие 95% ДИ, %": (100.0 * summary["coverage_h1"]).round(1),
            "Медиана времени, мс": summary["median_runtime_ms"].round(1),
            "Пик Python-памяти, МиБ": summary.get(
                "peak_python_memory_mib",
                pd.Series(float("nan"), index=summary.index),
            ).round(2),
            "Бэкенд": summary["backend"].replace(
                {
                    "pypi_hypex": "PyPI HypEx",
                    "native_compatible": "локальный fallback",
                    "native_reference_fallback": "локальный fallback",
                    "native": "локальный",
                }
            ),
        }
    )
    return result
