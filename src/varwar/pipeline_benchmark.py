from __future__ import annotations

import gc
import math
import time
import tracemalloc
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from varwar.benchmark import run_estimator_race
from varwar.core import ExperimentSpec
from varwar.estimators import (
    CalibrationWeightedATEEstimator,
    PopulationWeightedStratifiedEstimator,
    StratifiedDifferenceInMeansEstimator,
)
from varwar.pipelines import (
    PipelineCandidate,
    PipelineDefinition,
    PipelineTier,
    build_cartesian_rct_pipelines,
    build_full_traffic_pipelines,
    build_panel_prior_pipelines,
    build_stratified_pipelines,
)
from varwar.panel_prior import PanelPriorSummary
from varwar.stratification import (
    ABProportionalVarianceObjective,
    GeneticStrataReducer,
)
from varwar.synthetic import (
    SyntheticConfig,
    generate_synthetic_ab,
    generate_synthetic_panel,
)


@dataclass(frozen=True)
class SyntheticScenario:
    scenario_id: str
    name_ru: str
    outcome_model: str = "linear"
    heavy_tail: bool = False
    noise_distribution: str | None = None
    effect_heterogeneity: float = 0.0
    expert_safe_treatment_shift: float = 0.0
    gated_treatment_shift: float = 0.0
    population_stratum_weights: tuple[float, float, float] | None = None
    experiment_stratum_weights: tuple[float, float, float] | None = None
    stratum_effect_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0)
    purpose_ru: str = ""


SCENARIOS: dict[str, SyntheticScenario] = {
    "linear_gaussian": SyntheticScenario(
        "linear_gaussian",
        "Линейный сигнал, нормальный шум",
        purpose_ru="Благоприятный базовый сценарий.",
    ),
    "linear_heavy_tail": SyntheticScenario(
        "linear_heavy_tail",
        "Линейный сигнал, тяжёлые хвосты",
        heavy_tail=True,
        purpose_ru="Проверка устойчивости SE и выигрыша по дисперсии.",
    ),
    "nonlinear_gaussian": SyntheticScenario(
        "nonlinear_gaussian",
        "Нелинейный сигнал, нормальный шум",
        outcome_model="nonlinear",
        effect_heterogeneity=0.08,
        purpose_ru="Проверка линейных и квадратичных корректировок.",
    ),
    "contaminated_c": SyntheticScenario(
        "contaminated_c",
        "Treatment влияет на признак зоны C",
        gated_treatment_shift=0.45,
        purpose_ru="Проверка, что C-gate блокирует treatment-dependent признак.",
    ),
    "misclassified_b": SyntheticScenario(
        "misclassified_b",
        "Ошибочно безопасный признак зоны B",
        expert_safe_treatment_shift=0.30,
        purpose_ru="Отрицательный контроль: экспертная ошибка должна проявиться смещением.",
    ),
    "linear_skewed": SyntheticScenario(
        "linear_skewed",
        "Линейный сигнал, асимметричный логнормальный шум",
        noise_distribution="centered_lognormal",
        purpose_ru="Проверка асимметричных распределений и bootstrap inference.",
    ),
    "linear_contaminated": SyntheticScenario(
        "linear_contaminated",
        "Линейный сигнал, редкие сильные выбросы",
        noise_distribution="contaminated_normal",
        purpose_ru="Проверка робастных преобразований и устойчивости интервалов.",
    ),
    "population_shifted_strata": SyntheticScenario(
        "population_shifted_strata",
        "Нерепрезентативная выборка и известная целевая популяция",
        population_stratum_weights=(0.60, 0.30, 0.10),
        experiment_stratum_weights=(0.20, 0.40, 0.40),
        stratum_effect_offsets=(-0.10, 0.0, 0.60),
        purpose_ru=(
            "Отдельная честная гонка методов, использующих известные доли страт "
            "и population moments."
        ),
    ),
}


@dataclass(frozen=True)
class SyntheticPipelineRaceConfig:
    """Настройки многосценарной гонки полных analysis pipelines."""

    n_trials: int = 100
    n: int = 2_000
    alternative_effect: float = 0.15
    covariate_signal: float = 0.75
    noise_scale: float = 1.0
    seed: int = 20_260_724
    scenario_ids: tuple[str, ...] = (
        "linear_gaussian",
        "linear_heavy_tail",
        "nonlinear_gaussian",
        "contaminated_c",
        "misclassified_b",
        "population_shifted_strata",
    )
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
    include_stratified: bool = True
    heavy_tail: bool | None = None
    pipeline_ids: tuple[str, ...] | None = None
    information_mode: str = "both"
    panel_prior_months: int = 6
    include_non_rct_specialized: bool = False
    include_cartesian: bool = False

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive.")
        if self.n < 8:
            raise ValueError("n must be at least eight.")
        if not 0 < self.n_strata <= self.n:
            raise ValueError("n_strata must be in [1, n].")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between zero and one.")
        if self.information_mode not in {"experiment_only", "panel_prior", "both"}:
            raise ValueError(
                "information_mode must be 'experiment_only', 'panel_prior', or 'both'."
            )
        if self.panel_prior_months < 2:
            raise ValueError("panel_prior_months must be at least two.")
        unknown = set(self.scenario_ids).difference(SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown scenario IDs: {sorted(unknown)}")
        if self.pipeline_ids is not None:
            if not self.pipeline_ids:
                raise ValueError("pipeline_ids must not be empty.")
            if len(set(self.pipeline_ids)) != len(self.pipeline_ids):
                raise ValueError("pipeline_ids must be unique.")


def _configured_scenarios(
    config: SyntheticPipelineRaceConfig,
) -> tuple[SyntheticScenario, ...]:
    if config.heavy_tail is not None:
        scenario_id = "linear_heavy_tail" if config.heavy_tail else "linear_gaussian"
        return (SCENARIOS[scenario_id],)
    return tuple(SCENARIOS[scenario_id] for scenario_id in config.scenario_ids)


def _spec(config: SyntheticPipelineRaceConfig) -> ExperimentSpec:
    return ExperimentSpec(
        unit_col="unit_id",
        pre_target_col="y_pre",
        covariate_cols=("y_pre", "x1", "x2"),
        pre_treatment_cols=("y_pre", "x1", "x2"),
        expert_safe_in_time_cols=("calendar_index", "market_index"),
        gated_in_time_cols=("operational_load",),
        unsafe_demo_cols=("post_outcome_proxy",),
        alpha=config.alpha,
    )


def pipeline_definitions(
    config: SyntheticPipelineRaceConfig = SyntheticPipelineRaceConfig(),
) -> list[PipelineDefinition]:
    """Вернуть полный каталог строк текущей гонки."""

    spec = _spec(config)
    full = [
        candidate.definition
        for candidate in build_full_traffic_pipelines(
            spec,
            prefer_pypi_hypex=config.prefer_pypi_hypex,
        )
    ]
    stratified = []
    if config.include_stratified:
        stratified = [
            candidate.definition
            for candidate in build_stratified_pipelines(spec)
        ]
    if config.information_mode in {"panel_prior", "both"}:
        full.extend(
            candidate.definition
            for candidate in build_panel_prior_pipelines(
                spec,
                summary=None,
                prefer_pypi_hypex=config.prefer_pypi_hypex,
            )
        )
    definitions = full + stratified
    if config.include_cartesian:
        definitions.extend(
            candidate.definition
            for candidate in build_cartesian_rct_pipelines(
                spec,
                prefer_pypi_hypex=config.prefer_pypi_hypex,
                include_panel_prior=config.information_mode in {"panel_prior", "both"},
            )
        )
    if "population_shifted_strata" in config.scenario_ids and config.information_mode in {"panel_prior", "both"}:
        definitions += [
            candidate.definition
            for candidate in _population_information_candidates(
                SCENARIOS["population_shifted_strata"]
            )
        ]
    if not config.include_non_rct_specialized:
        definitions = [
            definition
            for definition in definitions
            if definition.tier != PipelineTier.CHANGED_METRIC
            and definition.pipeline_id != "variance_weighted_stratified"
        ]
    if config.pipeline_ids is None:
        return definitions
    requested = set(config.pipeline_ids)
    selected = [
        definition
        for definition in definitions
        if definition.pipeline_id in requested
    ]
    missing = requested.difference(
        definition.pipeline_id for definition in selected
    )
    if missing:
        raise ValueError(f"Unknown pipeline IDs: {sorted(missing)}")
    return selected


def _population_information_candidates(
    scenario: SyntheticScenario,
) -> list[PipelineCandidate]:
    weights_tuple = scenario.population_stratum_weights
    if weights_tuple is None:
        return []
    labels = ("low", "mid", "high")
    weights = dict(zip(labels, weights_tuple))
    population_score_mean = -weights_tuple[0] + weights_tuple[2]
    return [
        PipelineCandidate(
            PipelineDefinition(
                "stratified_mean_diff",
                "Стратификация с долями экспериментальной выборки",
                ("stratified_mean_diff",),
                "Sample ATE",
                PipelineTier.CONFIRMATORY,
                "Только A/B-датасет и population_stratum.",
                "Не использует известный состав целевой популяции.",
                "welch",
                estimand_id="sample_ate",
                information_mode="experiment_only",
            ),
            StratifiedDifferenceInMeansEstimator(),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "population_weighted_stratified",
                "Постстратификация по известной популяции",
                ("population_weighted_stratified",),
                "Population ATE",
                PipelineTier.CONFIRMATORY,
                "A/B-датасет и внешние доли population_stratum.",
                "Участвует только в population-information race.",
                "stratified_mean_diff",
                estimand_id="population_ate",
                information_mode="panel_prior",
            ),
            PopulationWeightedStratifiedEstimator(population_weights=weights),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "calibration_reweighting",
                "Entropy calibration по известным population moments",
                ("calibration_reweighting",),
                "Population ATE",
                PipelineTier.CONFIRMATORY,
                "A/B-датасет и внешний средний population_score.",
                "Каждая группа отдельно калибруется к целевой популяции.",
                "stratified_mean_diff",
                estimand_id="population_ate",
                information_mode="panel_prior",
            ),
            CalibrationWeightedATEEstimator(
                population_moments={"population_score": population_score_mean}
            ),
        ),
    ]


def _experiment_pair(
    config: SyntheticPipelineRaceConfig,
    scenario: SyntheticScenario,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    null = generate_synthetic_ab(
        SyntheticConfig(
            n=config.n,
            effect=0.0,
            covariate_signal=config.covariate_signal,
            noise_scale=config.noise_scale,
            seed=seed,
            heavy_tail=scenario.heavy_tail,
            noise_distribution=scenario.noise_distribution,
            outcome_model=scenario.outcome_model,
            effect_heterogeneity=0.0,
            expert_safe_treatment_shift=scenario.expert_safe_treatment_shift,
            gated_treatment_shift=scenario.gated_treatment_shift,
            experiment_stratum_weights=scenario.experiment_stratum_weights,
        )
    )
    alternative = null.copy()
    centered_pre = np.asarray(alternative["y_pre"], dtype=float)
    centered_pre = centered_pre - float(centered_pre.mean())
    individual_effect = (
        config.alternative_effect
        + scenario.effect_heterogeneity * centered_pre
    )
    if scenario.experiment_stratum_weights is not None:
        offsets = dict(
            zip(("low", "mid", "high"), scenario.stratum_effect_offsets)
        )
        individual_effect = individual_effect + np.asarray(
            alternative["population_stratum"].astype(object).map(offsets),
            dtype=float,
        )
    alternative["individual_effect"] = individual_effect
    alternative["y"] = (
        np.asarray(alternative["y0"], dtype=float)
        + np.asarray(alternative["treatment"], dtype=float) * individual_effect
    )
    return null, alternative


def _panel_prior_summary(
    config: SyntheticPipelineRaceConfig,
    scenario: SyntheticScenario,
    seed: int,
) -> PanelPriorSummary:
    """Build the SQL-like historical aggregate used by panel-prior pipelines."""

    panel = generate_synthetic_panel(
        SyntheticConfig(
            n=max(config.n // 2, 200),
            effect=0.0,
            covariate_signal=config.covariate_signal,
            noise_scale=config.noise_scale,
            seed=seed + 700_000,
            heavy_tail=scenario.heavy_tail,
            noise_distribution=scenario.noise_distribution,
            outcome_model=scenario.outcome_model,
            experiment_stratum_weights=scenario.experiment_stratum_weights,
        ),
        n_months=config.panel_prior_months,
        start_date="2023-01-01",
    )
    training_features = ("y_pre__lag2", "x1__lag2", "x2__lag2")
    return PanelPriorSummary.from_frame(
        panel,
        feature_cols=training_features,
        target_col="y_pre",
        date_col="report_date",
        cutoff="2024-01-01",
        entity_col="entity_id",
        category_cols=("region", "device", "activity_band"),
        stratum_col="population_stratum" if "population_stratum" in panel.columns else None,
        metadata={
            "source": "synthetic_monthly_panel",
            "scenario": scenario.scenario_id,
            "cutoff_rule": "report_date < experiment_start",
        },
    )


def _fit_reducer(
    data: pd.DataFrame,
    config: SyntheticPipelineRaceConfig,
    random_state: int,
    measure_memory: bool,
) -> tuple[GeneticStrataReducer, float, int | None]:
    reducer = GeneticStrataReducer(
        category_cols=config.category_cols,
        reference_target_col="y_pre",
        n_strata=config.n_strata,
        objective=ABProportionalVarianceObjective(),
        min_stratum_size=config.min_macro_stratum_size,
        population_size=config.population_size,
        max_generations=config.max_generations,
        elite_size=config.elite_size,
        mutation_probability=config.mutation_probability,
        patience=config.patience,
        random_state=random_state,
        unknown_policy="error",
        output_col="__varwar_ga_stratum",
    )
    start = time.perf_counter()
    if measure_memory:
        gc.collect()
        tracemalloc.start()
    try:
        reducer.fit(data)
    finally:
        if measure_memory:
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            peak_memory_bytes = None
    runtime_ms = (time.perf_counter() - start) * 1_000.0
    return reducer, runtime_ms, peak_memory_bytes


def _evaluate_candidates(
    data: pd.DataFrame,
    spec: ExperimentSpec,
    candidates: Iterable[PipelineCandidate],
    *,
    scenario: SyntheticScenario,
    hypothesis: str,
    true_effect: float,
    trial: int,
    seed: int,
    measure_memory: bool,
    preprocessor: str = "none",
    preprocessing_runtime_ms: float = 0.0,
    preprocessing_peak_memory_bytes: int | None = None,
    auxiliary_data: Any | None = None,
    race_information_mode: str = "experiment_only",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for candidate in candidates:
        frame = run_estimator_race(
            data,
            spec,
            estimators=[candidate.estimator],
            auxiliary_data=auxiliary_data,
            measure_memory=measure_memory,
        )
        for key, value in candidate.definition.to_record().items():
            frame[key] = value
        frame["data_scenario"] = scenario.scenario_id
        frame["scenario_name_ru"] = scenario.name_ru
        frame["scenario_purpose_ru"] = scenario.purpose_ru
        frame["hypothesis"] = hypothesis
        frame["true_effect"] = float(true_effect)
        frame["trial"] = int(trial)
        frame["seed"] = int(seed)
        frame["preprocessor"] = preprocessor
        frame["race_information_mode"] = race_information_mode
        frame["preprocessing_runtime_ms"] = float(preprocessing_runtime_ms)
        frame["end_to_end_runtime_ms"] = (
            pd.to_numeric(frame["runtime_ms"], errors="coerce")
            + float(preprocessing_runtime_ms)
        )
        estimator_peak = pd.to_numeric(
            frame.get("peak_memory_bytes", pd.Series(float("nan"), index=frame.index)),
            errors="coerce",
        )
        frame["preprocessing_peak_memory_bytes"] = preprocessing_peak_memory_bytes
        if preprocessing_peak_memory_bytes is None:
            frame["end_to_end_peak_memory_bytes"] = estimator_peak
        else:
            frame["end_to_end_peak_memory_bytes"] = np.maximum(
                estimator_peak.fillna(0.0).to_numpy(dtype=float),
                float(preprocessing_peak_memory_bytes),
            )
        frame["analysis_rows"] = (
            pd.to_numeric(frame.get("n_control"), errors="coerce")
            + pd.to_numeric(frame.get("n_treatment"), errors="coerce")
        )
        rows.append(frame)
    # Pandas 2.2 warns when concat has a column that is all-NA in only some
    # frames. Such columns carry no information in those frames, so omit them
    # before taking the union of result columns.
    compact_rows = [frame.dropna(axis=1, how="all") for frame in rows]
    return pd.concat(compact_rows, ignore_index=True, sort=False)


def run_synthetic_pipeline_race(
    config: SyntheticPipelineRaceConfig = SyntheticPipelineRaceConfig(),
    progress: Callable[[int, int], None] | None = None,
    *,
    measure_memory: bool = False,
) -> pd.DataFrame:
    """Запустить paired H0/H1 гонку на нескольких синтетических сценариях.

    Все обучаемые методы используют только доступный experiment dataset.
    CUPAC обучается на временном переходе lag2 -> lag1, GA использует
    pre-target, а B/C second-stage корректировки оцениваются по контролю.
    """

    spec = _spec(config)
    scenarios = _configured_scenarios(config)
    selected_ids = (
        set(config.pipeline_ids)
        if config.pipeline_ids is not None
        else None
    )
    stratified_ids = {
        candidate.definition.pipeline_id
        for candidate in build_stratified_pipelines(spec)
    }
    run_stratified = config.include_stratified and (
        selected_ids is None or bool(selected_ids.intersection(stratified_ids))
    )
    rows: list[pd.DataFrame] = []
    information_modes = (
        ("experiment_only", "panel_prior")
        if config.information_mode == "both"
        else (config.information_mode,)
    )
    total = config.n_trials * len(scenarios) * len(information_modes)
    completed = 0

    for scenario_index, scenario in enumerate(scenarios):
        for trial in range(config.n_trials):
            seed = config.seed + scenario_index * 1_000_000 + trial
            null, alternative = _experiment_pair(config, scenario, seed)
            population_true_effect = config.alternative_effect
            if scenario.population_stratum_weights is not None:
                population_true_effect += float(
                    np.dot(
                        scenario.population_stratum_weights,
                        scenario.stratum_effect_offsets,
                    )
                )
            for race_information_mode in information_modes:
                panel_summary = (
                    _panel_prior_summary(config, scenario, seed)
                    if race_information_mode == "panel_prior"
                    else None
                )
                full_candidates = [
                    candidate
                    for candidate in build_full_traffic_pipelines(
                        spec,
                        prefer_pypi_hypex=config.prefer_pypi_hypex,
                        include_specialized=False,
                    )
                    if candidate.definition.tier != PipelineTier.CHANGED_METRIC
                ]
                if panel_summary is not None:
                    full_candidates.extend(
                        build_panel_prior_pipelines(
                            spec,
                            panel_summary,
                            prefer_pypi_hypex=config.prefer_pypi_hypex,
                        )
                    )
                if config.include_cartesian:
                    full_candidates.extend(
                        candidate
                        for candidate in build_cartesian_rct_pipelines(
                            spec,
                            prefer_pypi_hypex=config.prefer_pypi_hypex,
                            include_panel_prior=panel_summary is not None,
                            panel_summary=panel_summary,
                        )
                        if candidate.definition.information_mode == "experiment_only"
                        or race_information_mode == "panel_prior"
                    )
                if selected_ids is not None:
                    full_candidates = [
                        candidate
                        for candidate in full_candidates
                        if candidate.definition.pipeline_id in selected_ids
                    ]
                reducer = None
                ga_runtime = 0.0
                ga_peak = None
                if run_stratified:
                    reducer, ga_runtime, ga_peak = _fit_reducer(
                        null,
                        config,
                        random_state=seed + 100_000,
                        measure_memory=measure_memory,
                    )
                for hypothesis, true_effect, experiment in (
                    ("H0", 0.0, null),
                    ("H1", population_true_effect, alternative),
                ):
                    if full_candidates:
                        rows.append(
                            _evaluate_candidates(
                                experiment,
                                spec,
                                full_candidates,
                                scenario=scenario,
                                hypothesis=hypothesis,
                                true_effect=true_effect,
                                trial=trial,
                                seed=seed,
                                measure_memory=measure_memory,
                                auxiliary_data=panel_summary,
                                race_information_mode=race_information_mode,
                            )
                        )
                    if run_stratified and reducer is not None:
                        prepared = reducer.augment(experiment)
                        stratified_candidates = [
                            candidate
                            for candidate in build_stratified_pipelines(
                                spec,
                                prefer_pypi_hypex=config.prefer_pypi_hypex,
                            )
                            if candidate.definition.pipeline_id != "variance_weighted_stratified"
                        ]
                        if selected_ids is not None:
                            stratified_candidates = [
                                candidate
                                for candidate in stratified_candidates
                                if candidate.definition.pipeline_id in selected_ids
                            ]
                        if stratified_candidates:
                            rows.append(
                                _evaluate_candidates(
                                    prepared,
                                    spec.with_stratum(reducer.output_col),
                                    stratified_candidates,
                                    scenario=scenario,
                                    hypothesis=hypothesis,
                                    true_effect=true_effect,
                                    trial=trial,
                                    seed=seed,
                                    measure_memory=measure_memory,
                                    preprocessor="genetic_strata_reducer",
                                    preprocessing_runtime_ms=ga_runtime,
                                    preprocessing_peak_memory_bytes=ga_peak,
                                    auxiliary_data=panel_summary,
                                    race_information_mode=race_information_mode,
                                )
                            )
                    if race_information_mode == "panel_prior":
                        population_candidates = _population_information_candidates(scenario)
                        if selected_ids is not None:
                            population_candidates = [
                                candidate
                                for candidate in population_candidates
                                if candidate.definition.pipeline_id in selected_ids
                            ]
                        if population_candidates:
                            rows.append(
                                _evaluate_candidates(
                                    experiment,
                                    spec.with_stratum("population_stratum"),
                                    population_candidates,
                                    scenario=scenario,
                                    hypothesis=hypothesis,
                                    true_effect=true_effect,
                                    trial=trial,
                                    seed=seed,
                                    measure_memory=measure_memory,
                                    preprocessor="known_population_information",
                                    auxiliary_data=panel_summary,
                                    race_information_mode=race_information_mode,
                                )
                            )
                completed += 1
                if progress is not None:
                    progress(completed, total)
    if not rows:
        raise ValueError("No pipelines were selected for the synthetic race.")
    return pd.concat(rows, ignore_index=True, sort=False)


def profile_synthetic_pipeline_memory(
    config: SyntheticPipelineRaceConfig = SyntheticPipelineRaceConfig(),
) -> pd.DataFrame:
    """Измерить Python allocations в отдельном steady-state прогоне."""

    first_scenario = _configured_scenarios(config)[0].scenario_id
    profile_config = replace(
        config,
        n_trials=1,
        scenario_ids=(first_scenario,),
        heavy_tail=None,
    )
    run_synthetic_pipeline_race(profile_config, measure_memory=False)
    profile = run_synthetic_pipeline_race(profile_config, measure_memory=True)
    ok = profile.loc[profile["status"] == "ok"].copy()
    keys = ["data_scenario", "pipeline"]
    result = (
        ok.groupby(keys, sort=False, dropna=False)["end_to_end_peak_memory_bytes"]
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


def _binomial_calibration_pvalue(
    successes: int,
    trials: int,
    probability: float,
) -> float:
    if trials < 1:
        return math.nan
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(scipy_stats.binomtest(successes, trials, probability).pvalue)
    except Exception:
        variance = trials * probability * (1.0 - probability)
        if variance <= 0.0:
            return math.nan
        z_value = abs(successes - trials * probability) / math.sqrt(variance)
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


def _mean_zero_pvalue(mean: float, empirical_sd: float, n: int) -> float:
    if n < 2 or not math.isfinite(empirical_sd) or empirical_sd <= 0.0:
        return math.nan
    statistic = abs(mean) / (empirical_sd / math.sqrt(n))
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(2.0 * scipy_stats.t.sf(statistic, n - 1))
    except Exception:
        return math.erfc(statistic / math.sqrt(2.0))


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
    coverage_mask = np.isfinite(ci_low) & np.isfinite(ci_high)
    covered = coverage_mask & (ci_low <= true_effect) & (true_effect <= ci_high)
    runtime = np.asarray(
        pd.to_numeric(ok.get("end_to_end_runtime_ms"), errors="coerce"),
        dtype=float,
    )
    analysis_rows = np.asarray(
        pd.to_numeric(ok.get("analysis_rows"), errors="coerce"),
        dtype=float,
    )
    backends = sorted(
        {
            str(value)
            for value in ok.get("diag_backend", pd.Series(dtype=object)).dropna()
        }
    )
    rejected_c = ok.get(
        "diag_transformer_zone_c_rejected",
        pd.Series("", index=ok.index, dtype=object),
    ).fillna("")
    mean_effect = float(effect.mean()) if effect.size else math.nan
    empirical_sd = (
        float(effect.std(ddof=1)) if effect.size > 1 else math.nan
    )
    bias = mean_effect - true_effect if effect.size else math.nan
    return {
        "n_total": int(len(frame)),
        "n_success": int(effect.size),
        "success_rate": float(effect.size / len(frame)) if len(frame) else math.nan,
        "mean_effect": mean_effect,
        "bias": bias,
        "bias_p_value": _mean_zero_pvalue(bias, empirical_sd, int(effect.size)),
        "empirical_sd": empirical_sd,
        "mean_se": float(se.mean()) if se.size else math.nan,
        "rejection_rate": float(np.mean(p_value < alpha)) if p_value.size else math.nan,
        "rejection_count": int(np.sum(p_value < alpha)),
        "coverage": (
            float(covered.sum() / coverage_mask.sum())
            if coverage_mask.any()
            else math.nan
        ),
        "coverage_count": int(covered.sum()),
        "coverage_trials": int(coverage_mask.sum()),
        "median_runtime_ms": (
            float(np.nanmedian(runtime)) if np.isfinite(runtime).any() else math.nan
        ),
        "median_analysis_rows": (
            float(np.nanmedian(analysis_rows))
            if np.isfinite(analysis_rows).any()
            else math.nan
        ),
        "backend": ",".join(backends) if backends else "native",
        "zone_c_rejection_rate": (
            float(np.mean(rejected_c.astype(str).str.len() > 0))
            if len(rejected_c)
            else math.nan
        ),
    }


def summarize_synthetic_pipeline_race(
    trials: pd.DataFrame,
    alpha: float = 0.05,
    *,
    max_abs_bias: float = 0.02,
    max_false_positive_rate: float = 0.08,
    min_coverage: float = 0.92,
    min_se_ratio: float = 0.80,
    max_se_ratio: float = 1.20,
    min_trials_for_scientific_ranking: int = 100,
) -> pd.DataFrame:
    """Свести paired H0/H1 прогоны и применить практические R&D-допуски.

    Допуски являются прозрачным диагностическим экраном, а не доказательством
    статистической валидности или результатом множественной проверки.
    """

    required = {
        "data_scenario",
        "pipeline",
        "hypothesis",
        "true_effect",
        "status",
    }
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"Missing pipeline-race columns: {sorted(missing)}")

    metadata_columns = [
        "scenario_name_ru",
        "scenario_purpose_ru",
        "pipeline_name_ru",
        "components",
        "estimand",
        "pipeline_tier",
        "data_contract",
        "pipeline_notes",
        "predecessor_pipeline",
        "design_mode",
        "information_mode",
        "estimand_id",
        "temporal_mode",
        "race_information_mode",
    ]
    grouping_columns = ["data_scenario", "pipeline"] + [
        column
        for column in (
            "race_information_mode",
            "information_mode",
            "temporal_mode",
            "estimand_id",
        )
        if column in trials.columns
    ]
    rows: list[dict[str, Any]] = []
    for group_key, frame in trials.groupby(
        grouping_columns,
        sort=False,
        dropna=False,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(grouping_columns, group_key))
        scenario_id = group_values["data_scenario"]
        pipeline_id = group_values["pipeline"]
        null = frame.loc[frame["hypothesis"] == "H0"]
        alternative = frame.loc[frame["hypothesis"] == "H1"]
        if null.empty or alternative.empty:
            continue
        h0 = _hypothesis_metrics(null, 0.0, alpha)
        true_effect = float(alternative["true_effect"].iloc[0])
        h1 = _hypothesis_metrics(alternative, true_effect, alpha)
        fpr_low, fpr_high = _binomial_interval(
            h0["rejection_count"],
            h0["n_success"],
        )
        power_low, power_high = _binomial_interval(
            h1["rejection_count"],
            h1["n_success"],
        )
        fpr_calibration_p = _binomial_calibration_pvalue(
            h0["rejection_count"],
            h0["n_success"],
            alpha,
        )
        coverage_calibration_p = _binomial_calibration_pvalue(
            h1["coverage_count"],
            h1["coverage_trials"],
            1.0 - alpha,
        )
        first = frame.iloc[0]
        se_ratio = (
            h1["mean_se"] / h1["empirical_sd"]
            if h1["empirical_sd"] > 0.0
            else math.nan
        )
        record: dict[str, Any] = {
            "data_scenario": scenario_id,
            "pipeline": pipeline_id,
            "n_trials": min(h0["n_total"], h1["n_total"]),
            "success_rate": min(h0["success_rate"], h1["success_rate"]),
            "analysis_rows": h1["median_analysis_rows"],
            "true_effect": true_effect,
            "mean_effect_h1": h1["mean_effect"],
            "bias_h0": h0["bias"],
            "bias_h1": h1["bias"],
            "bias_h1_p_value": h1["bias_p_value"],
            "bias_calibrated": bool(
                math.isfinite(h1["bias_p_value"])
                and h1["bias_p_value"] >= alpha
            ),
            "bias_within_tolerance": bool(
                math.isfinite(h1["bias"])
                and abs(h1["bias"]) <= max_abs_bias
            ),
            "empirical_sd_h1": h1["empirical_sd"],
            "mean_se_h1": h1["mean_se"],
            "se_to_empirical_sd": se_ratio,
            "se_ratio_within_tolerance": bool(
                math.isfinite(se_ratio)
                and min_se_ratio <= se_ratio <= max_se_ratio
            ),
            "false_positive_rate": h0["rejection_rate"],
            "fpr_ci_low": fpr_low,
            "fpr_ci_high": fpr_high,
            "fpr_calibration_p_value": fpr_calibration_p,
            "fpr_calibrated": bool(fpr_calibration_p >= alpha),
            "fpr_within_tolerance": bool(
                math.isfinite(h0["rejection_rate"])
                and h0["rejection_rate"] <= max_false_positive_rate
            ),
            "power": h1["rejection_rate"],
            "power_ci_low": power_low,
            "power_ci_high": power_high,
            "coverage_h1": h1["coverage"],
            "coverage_calibration_p_value": coverage_calibration_p,
            "coverage_calibrated": bool(coverage_calibration_p >= alpha),
            "coverage_within_tolerance": bool(
                math.isfinite(h1["coverage"])
                and h1["coverage"] >= min_coverage
            ),
            "max_abs_bias_tolerance": max_abs_bias,
            "max_fpr_tolerance": max_false_positive_rate,
            "min_coverage_tolerance": min_coverage,
            "min_se_ratio_tolerance": min_se_ratio,
            "max_se_ratio_tolerance": max_se_ratio,
            "median_runtime_ms": float(
                np.nanmedian([h0["median_runtime_ms"], h1["median_runtime_ms"]])
            ),
            "backend": h1["backend"],
            "zone_c_rejection_rate": float(
                np.nanmax(
                    [h0["zone_c_rejection_rate"], h1["zone_c_rejection_rate"]]
                )
            ),
        }
        record.update({column: first.get(column) for column in metadata_columns})
        record.update(group_values)
        rows.append(record)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Historical fixtures and old result files did not carry the explicit
    # estimand_id column. Infer only the narrow sample-ATE fallback so those
    # files retain their diagnostic meaning; current runs always provide the
    # field from PipelineDefinition.
    if "estimand_id" in summary.columns:
        missing_estimand = summary["estimand_id"].isna()
        inferred = summary["estimand"].astype(str).str.contains(
            r"\bATE\b", regex=True
        )
        summary.loc[missing_estimand & inferred, "estimand_id"] = "sample_ate"
        summary.loc[missing_estimand & ~inferred, "estimand_id"] = "unknown"
    summary["variance_reduction_vs_welch"] = math.nan
    summary["variance_reduction_vs_cupac_a"] = math.nan
    summary["incremental_variance_reduction"] = math.nan
    efficiency_grouping = ["data_scenario"] + [
        column
        for column in (
            "race_information_mode",
            "information_mode",
            "temporal_mode",
            "estimand_id",
        )
        if column in summary.columns
    ]
    for _, indices in summary.groupby(efficiency_grouping, sort=False, dropna=False).groups.items():
        block = summary.loc[indices]
        _fill_relative_efficiency(
            summary,
            indices,
            block,
            baseline_pipeline="welch",
            output_column="variance_reduction_vs_welch",
        )
        _fill_relative_efficiency(
            summary,
            indices,
            block,
            baseline_pipeline="cupac_linear_A__hypex_ab",
            output_column="variance_reduction_vs_cupac_a",
        )
        for index in indices:
            predecessor = summary.at[index, "predecessor_pipeline"]
            if not isinstance(predecessor, str) or not predecessor:
                continue
            baseline = block.loc[
                block["pipeline"] == predecessor,
                "empirical_sd_h1",
            ]
            if baseline.empty:
                continue
            base_variance = float(baseline.iloc[0]) ** 2
            method_variance = float(summary.at[index, "empirical_sd_h1"]) ** 2
            if base_variance > 0.0 and math.isfinite(method_variance):
                summary.at[index, "incremental_variance_reduction"] = (
                    1.0 - method_variance / base_variance
                )

    changed_metric = (
        summary["pipeline_tier"] == PipelineTier.CHANGED_METRIC.value
    )
    summary.loc[
        changed_metric,
        [
            "true_effect",
            "bias_h0",
            "bias_h1",
            "bias_h1_p_value",
            "coverage_h1",
            "coverage_calibration_p_value",
        ],
    ] = math.nan
    summary.loc[
        changed_metric,
        [
            "bias_calibrated",
            "bias_within_tolerance",
            "coverage_calibrated",
            "coverage_within_tolerance",
        ],
    ] = False

    comparable = (
        summary["estimand_id"].astype(str).eq("sample_ate")
        if "estimand_id" in summary.columns
        else summary["estimand"].astype(str).str.contains(r"\bATE\b", regex=True)
    )
    comparable &= ~changed_metric
    summary["passes_diagnostic_screen"] = (
        comparable
        & summary["bias_within_tolerance"]
        & summary["fpr_within_tolerance"]
        & summary["coverage_within_tolerance"]
        & summary["se_ratio_within_tolerance"]
        & (summary["success_rate"] == 1.0)
    )
    summary["scientific_sample_size_ok"] = (
        summary["n_trials"] >= int(min_trials_for_scientific_ranking)
    )
    summary["ranking_eligible"] = (
        summary["passes_diagnostic_screen"]
        & summary["scientific_sample_size_ok"]
    )
    summary["diagnostic_status"] = np.select(
        [
            ~comparable,
            summary["success_rate"] < 1.0,
            ~summary["bias_within_tolerance"],
            ~summary["fpr_within_tolerance"],
            ~summary["coverage_within_tolerance"],
            ~summary["se_ratio_within_tolerance"],
        ],
        [
            "другой estimand",
            "ошибки выполнения",
            "смещение выше допуска",
            "FPR выше допуска",
            "покрытие ниже допуска",
            "SE/SD вне допуска",
        ],
        default="прошёл диагностический допуск",
    )
    return summary.sort_values(
        ["data_scenario", "passes_diagnostic_screen", "empirical_sd_h1"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _fill_relative_efficiency(
    target: pd.DataFrame,
    indices: Iterable[int],
    block: pd.DataFrame,
    *,
    baseline_pipeline: str,
    output_column: str,
) -> None:
    baseline = block.loc[
        block["pipeline"] == baseline_pipeline,
        "empirical_sd_h1",
    ]
    if baseline.empty:
        return
    baseline_variance = float(baseline.iloc[0]) ** 2
    if baseline_variance <= 0.0:
        return
    target.loc[list(indices), output_column] = 1.0 - (
        target.loc[list(indices), "empirical_sd_h1"] ** 2 / baseline_variance
    )


def format_pipeline_summary_ru(summary: pd.DataFrame) -> pd.DataFrame:
    """Подготовить понятную русскую таблицу для отчёта."""

    def interval(value: float, low: float, high: float) -> str:
        return f"{100.0 * value:.1f} [{100.0 * low:.1f}; {100.0 * high:.1f}]"

    return pd.DataFrame(
        {
            "Сценарий": summary["scenario_name_ru"],
            "Пайплайн": summary["pipeline_name_ru"],
            "Контур": summary["pipeline_tier"].map(
                {
                    "confirmatory": "подтверждающий",
                    "integration_parity": "сверка HypEx",
                    "exploratory": "исследовательский",
                    "changed_metric": "другая метрика",
                }
            ).fillna(summary["pipeline_tier"]),
            "Диагностический статус": summary["diagnostic_status"],
            "Оценка эффекта": summary["mean_effect_h1"].round(4),
            "Смещение": summary["bias_h1"].round(4),
            "Эмпирическое SD": summary["empirical_sd_h1"].round(4),
            "Средний SE": summary["mean_se_h1"].round(4),
            "SE / SD": summary["se_to_empirical_sd"].round(3),
            "Снижение дисперсии к Welch, %": (
                100.0 * summary["variance_reduction_vs_welch"]
            ).round(1),
            "Снижение дисперсии к CUPAC A, %": (
                100.0 * summary["variance_reduction_vs_cupac_a"]
            ).round(1),
            "Инкремент к predecessor, %": (
                100.0 * summary["incremental_variance_reduction"]
            ).round(1),
            "FPR, % [95% ДИ]": [
                interval(value, low, high)
                for value, low, high in zip(
                    summary["false_positive_rate"],
                    summary["fpr_ci_low"],
                    summary["fpr_ci_high"],
                )
            ],
            "Мощность, % [95% ДИ]": [
                interval(value, low, high)
                for value, low, high in zip(
                    summary["power"],
                    summary["power_ci_low"],
                    summary["power_ci_high"],
                )
            ],
            "Покрытие 95% ДИ, %": (100.0 * summary["coverage_h1"]).round(1),
            "C-gate отклонил признак, %": (
                100.0 * summary["zone_c_rejection_rate"]
            ).round(1),
            "Медиана времени, мс": summary["median_runtime_ms"].round(2),
            "Пик Python-памяти, МиБ": summary.get(
                "peak_python_memory_mib",
                pd.Series(float("nan"), index=summary.index),
            ).round(2),
            "Режим информации гонки": summary.get(
                "race_information_mode",
                pd.Series("experiment_only", index=summary.index),
            ),
            "Временной доступ": summary.get(
                "temporal_mode",
                pd.Series("pre_treatment_only", index=summary.index),
            ),
            "Идентификатор estimand": summary.get("estimand_id", summary["estimand"]),
            "Достаточно повторений для рейтинга": summary.get(
                "scientific_sample_size_ok",
                pd.Series(False, index=summary.index),
            ),
            "Допущен в научный рейтинг": summary.get(
                "ranking_eligible",
                pd.Series(False, index=summary.index),
            ),
            "Backend": summary["backend"].replace(
                {
                    "pypi_hypex": "PyPI HypEx",
                    "native_compatible": "локальный fallback",
                    "native_reference_fallback": "локальный fallback",
                    "native": "локальный",
                }
            ),
        }
    )
