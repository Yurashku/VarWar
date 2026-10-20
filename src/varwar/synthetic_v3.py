from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd


_OUTCOME_COLUMNS = frozenset(
    {
        "treatment",
        "assigned",
        "received",
        "outcome",
        "target",
        "y",
        "y0",
        "y1",
        "individual_effect",
        "zone_b0",
        "zone_b1",
        "zone_c0",
        "zone_c1",
        "post_outcome_proxy0",
        "post_outcome_proxy1",
    }
)


@dataclass(frozen=True)
class SyntheticWorldSpec:
    """Полная декларация одного синтетического мира без назначения treatment."""

    world_id: str
    name_ru: str
    purpose_ru: str
    n_experiment: int = 2_000
    population_size: int = 10_000
    treatment_share: float = 0.5
    n_pre_periods: int = 6
    r2: float = 0.4
    effect: float = 0.2
    outcome_distribution: str = "gaussian"
    signal_model: str = "linear"
    heteroskedastic: bool = False
    heterogeneous_effect: bool = False
    n_irrelevant_features: int = 2
    high_dimensional: bool = False
    cohort_shift: float = 0.0
    panel_shift: float = 0.0
    temporal_drift: float = 0.0
    prior_mismatch: float = 0.0
    zone_b_mode: str = "safe"
    zone_c_mode: str = "safe"
    many_rare_strata: bool = False
    effect_modifier_centering: str = "population"
    outcome_process: str = "static"
    unit_effect_share: float = 0.0
    effect_modifier_source: str = "x1_x3"
    effect_modifier_scale: float = 0.35
    treated_noise_scale: float = 1.0
    noise_in_covariates: bool = False

    def __post_init__(self) -> None:
        if not self.world_id:
            raise ValueError("world_id must not be empty.")
        if self.n_experiment < 20:
            raise ValueError("n_experiment must be at least 20.")
        if self.population_size < self.n_experiment:
            raise ValueError("population_size must be at least n_experiment.")
        if self.n_pre_periods < 2:
            raise ValueError("n_pre_periods must be at least two.")
        if not 0.0 < self.treatment_share < 1.0:
            raise ValueError("treatment_share must be in (0, 1).")
        if not 0.0 <= self.r2 < 1.0:
            raise ValueError("r2 must be in [0, 1).")
        if self.outcome_distribution not in {
            "gaussian",
            "student_t3",
            "lognormal",
            "contaminated",
            "zero_inflated_count",
        }:
            raise ValueError("Unknown outcome_distribution.")
        if self.signal_model not in {"linear", "quadratic", "threshold", "interactions"}:
            raise ValueError("Unknown signal_model.")
        if self.zone_b_mode not in {"safe", "treatment_dependent", "outcome_proxy"}:
            raise ValueError("Unknown zone_b_mode.")
        if self.zone_c_mode not in {"safe", "treatment_dependent", "outcome_proxy"}:
            raise ValueError("Unknown zone_c_mode.")
        if self.effect_modifier_centering not in {"population", "cohort"}:
            raise ValueError("effect_modifier_centering must be 'population' or 'cohort'.")
        if self.outcome_process not in {"static", "panel"}:
            raise ValueError("outcome_process must be 'static' or 'panel'.")
        if self.effect_modifier_source not in {"x1_x3", "prognostic"}:
            raise ValueError("effect_modifier_source must be 'x1_x3' or 'prognostic'.")
        if not math.isfinite(self.effect_modifier_scale):
            raise ValueError("effect_modifier_scale must be finite.")
        if not (math.isfinite(self.treated_noise_scale) and self.treated_noise_scale > 0.0):
            raise ValueError("treated_noise_scale must be positive.")
        if not 0.0 <= self.unit_effect_share < 1.0:
            raise ValueError("unit_effect_share must be in [0, 1).")
        if self.outcome_process == "static" and self.unit_effect_share != 0.0:
            raise ValueError("unit_effect_share requires outcome_process='panel'.")
        if self.outcome_process == "panel":
            if self.r2 + self.unit_effect_share >= 1.0:
                raise ValueError("Panel worlds require r2 + unit_effect_share < 1.")
            if self.outcome_distribution == "zero_inflated_count":
                raise ValueError("Panel worlds do not support zero_inflated_count.")
            if self.temporal_drift != 0.0:
                raise ValueError("Panel worlds keep the series stationary; temporal_drift must be 0.")
            if self.population_size <= self.n_experiment:
                raise ValueError("Panel worlds need entities outside the cohort for the external panel.")
        if self.outcome_distribution == "zero_inflated_count" and self.treated_noise_scale != 1.0:
            raise ValueError("treated_noise_scale is not defined for zero_inflated_count.")
        if self.noise_in_covariates and max(
            self.n_irrelevant_features, 24 if self.high_dimensional else 0
        ) == 0:
            raise ValueError("noise_in_covariates requires irrelevant noise features.")


@dataclass(frozen=True)
class NativeEstimandTruth:
    """Точное значение одного estimand в популяции и экспериментальной когорте."""

    estimand_id: str
    sample_value: float
    population_value: float
    description_ru: str


@dataclass(frozen=True)
class SyntheticWorld:
    """Данные мира до назначения и точные истины конечной популяции."""

    spec: SyntheticWorldSpec
    finite_population: pd.DataFrame
    pre_period_panel: pd.DataFrame
    experiment_cohort: pd.DataFrame
    assignment_data: pd.DataFrame
    potential_outcomes: pd.DataFrame
    true_sample_ate: float
    true_population_ate: float
    native_truths: Mapping[str, NativeEstimandTruth]
    estimand_truths: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _world(
    world_id: str,
    name_ru: str,
    purpose_ru: str,
    **changes: object,
) -> SyntheticWorldSpec:
    return SyntheticWorldSpec(
        world_id=world_id,
        name_ru=name_ru,
        purpose_ru=purpose_ru,
        **changes,
    )


_CATALOG_ITEMS = (
    *(
        _world(
            f"gaussian_r2_{str(r2).replace('.', '')}_n{n}",
            f"Gaussian, R2={r2}, n={n}",
            "Базовая проверка калибровки и выигрыша от прогностических признаков.",
            n_experiment=n,
            population_size=max(5_000, 5 * n),
            r2=r2,
        )
        for n in (200, 2_000)
        for r2 in (0.0, 0.4, 0.8)
    ),
    _world(
        "gaussian_30_70",
        "Gaussian, назначение 30/70",
        "Метаданные для проверки несимметричных механизмов назначения.",
        treatment_share=0.3,
    ),
    _world("student_t3", "Student-t(3)", "Тяжёлые хвосты при конечной дисперсии.", outcome_distribution="student_t3"),
    _world("lognormal", "Логнормальный outcome", "Сильная асимметрия outcome.", outcome_distribution="lognormal"),
    _world("contaminated", "Загрязнённая нормаль", "Редкие крупные выбросы.", outcome_distribution="contaminated"),
    _world(
        "zero_inflated_count",
        "Zero-inflated count",
        "Дискретный outcome, много нулей и совпадающих значений.",
        outcome_distribution="zero_inflated_count",
    ),
    _world("heteroskedastic", "Гетероскедастичность", "Дисперсия зависит от признаков.", heteroskedastic=True),
    _world("quadratic_signal", "Квадратичный сигнал", "Проверка polynomial CUPAC.", signal_model="quadratic"),
    _world("threshold_signal", "Пороговый сигнал", "Проверка нелинейных моделей.", signal_model="threshold"),
    _world(
        "interactions_high_dimensional",
        "Взаимодействия и лишние признаки",
        "Проверка Auto-ML и устойчивости к нерелевантным признакам.",
        signal_model="interactions",
        high_dimensional=True,
        n_irrelevant_features=24,
    ),
    _world("heterogeneous_effect", "Неоднородный эффект", "Различие sample и subgroup effects.", heterogeneous_effect=True),
    _world("population_shift", "Смещение экспериментальной когорты", "Различие sample ATE и population ATE.", cohort_shift=1.25, heterogeneous_effect=True),
    _world("panel_shift", "Смещение исторической панели", "Проверка переноса panel prior.", panel_shift=1.0),
    _world("temporal_drift", "Временной дрейф", "Изменение связи признаков с target по месяцам.", temporal_drift=0.7),
    _world("strong_prior_mismatch", "Сильный prior mismatch", "Проверка отказа от несовместимого prior.", panel_shift=1.0, prior_mismatch=1.5),
    _world("temporal_bc_safe", "Безопасные зоны B и C", "Потенциальные B/C не зависят от treatment."),
    _world("temporal_b_unsafe", "Ошибочно безопасная зона B", "B является treatment-dependent признаком.", zone_b_mode="treatment_dependent"),
    _world("temporal_c_unsafe", "Опасная зона C", "C является post-outcome proxy.", zone_c_mode="outcome_proxy"),
    _world("many_rare_strata", "Много редких атомарных страт", "Стресс-тест построения и укрупнения страт.", many_rare_strata=True),
    # --- v3.1: ярус n=500 и честные high-dimensional ковариаты ---------------
    _world(
        "gaussian_r2_04_n500",
        "Gaussian, R2=0.4, n=500",
        "Промежуточный объём выборки между n=200 и n=2000.",
        n_experiment=500,
        population_size=5_000,
    ),
    _world(
        "heteroskedastic_n500",
        "Гетероскедастичность, n=500",
        "HC2 против классической SE при умеренном n.",
        n_experiment=500,
        population_size=5_000,
        heteroskedastic=True,
    ),
    _world(
        "high_dimensional_covariates_n500",
        "24 шумовых ковариаты в анализе, n=500",
        "noise_* реально входят в covariate_cols: Lin платит 2K+2 параметра.",
        n_experiment=500,
        population_size=5_000,
        n_irrelevant_features=24,
        high_dimensional=True,
        noise_in_covariates=True,
    ),
    # --- v3.1: единый панельный процесс Y_it = alpha_i + f(X_i) + d_t + eps_it -
    *(
        _world(
            f"gaussian_n{n}_panel",
            f"Панель, Gaussian, n={n}",
            "Outcome — следующий период того же ряда; CUPED/CUPAC не ущемлены DGP.",
            outcome_process="panel",
            n_experiment=n,
            population_size=max(5_000, 5 * n),
            r2=0.2,
            unit_effect_share=0.4,
        )
        for n in (200, 500, 2_000)
    ),
    _world(
        "heterogeneous_effect_panel",
        "Панель, неоднородный эффект",
        "Неоднородный эффект при персистентном unit-эффекте.",
        outcome_process="panel",
        r2=0.2,
        unit_effect_share=0.4,
        heterogeneous_effect=True,
    ),
    _world(
        "population_shift_panel",
        "Панель, смещение когорты",
        "SATE и PATE различаются; H0 корректна при centering по estimand.",
        outcome_process="panel",
        r2=0.2,
        unit_effect_share=0.4,
        cohort_shift=1.25,
        heterogeneous_effect=True,
    ),
    _world(
        "external_shift_panel",
        "Панель, сдвиг только внешней панели",
        "Panel prior смещён, собственный pre-period эксперимента — нет.",
        outcome_process="panel",
        r2=0.2,
        unit_effect_share=0.4,
        panel_shift=1.0,
    ),
    _world(
        "external_prior_mismatch_panel",
        "Панель, несовместимый внешний prior",
        "Prior из внешней панели противоречит эксперименту; его нужно отбросить.",
        outcome_process="panel",
        r2=0.2,
        unit_effect_share=0.4,
        panel_shift=1.0,
        prior_mismatch=1.5,
    ),
    _world(
        "nonlinear_panel",
        "Панель, квадратичный сигнал признаков",
        "Нелинейный f(X): cross-fitted GBM должен обогнать линейный Lin.",
        outcome_process="panel",
        signal_model="quadratic",
        r2=0.7,
        unit_effect_share=0.05,
    ),
    *(
        _world(
            f"freedman_p{int(round(100 * share))}_panel",
            f"Панель, разбиение {int(round(100 * share))}/{int(round(100 * (1 - share)))}, неоднородный эффект",
            "Случай Freedman: pooled ANCOVA/CUPED может проиграть DIM, Lin — нет.",
            outcome_process="panel",
            treatment_share=share,
            r2=0.25,
            unit_effect_share=0.35,
            heterogeneous_effect=True,
            effect_modifier_source="prognostic",
            effect_modifier_scale=-1.2,
            treated_noise_scale=1.5,
        )
        for share in (0.2, 0.3)
    ),
)

WORLD_CATALOG: Mapping[str, SyntheticWorldSpec] = MappingProxyType(
    {item.world_id: item for item in _CATALOG_ITEMS}
)


def list_worlds() -> pd.DataFrame:
    """Вернуть компактный каталог миров, пригодный для отчёта."""

    return pd.DataFrame([vars(spec) for spec in WORLD_CATALOG.values()])


def get_world_spec(world_id: str, **overrides: object) -> SyntheticWorldSpec:
    """Получить спецификацию каталога и, при необходимости, заменить параметры."""

    try:
        spec = WORLD_CATALOG[world_id]
    except KeyError as error:
        raise KeyError(f"Unknown synthetic world: {world_id!r}.") from error
    return replace(spec, **overrides) if overrides else spec


BASE_PRE_TREATMENT_COLS = ("pre_target_lag1", "x1", "x2", "x3", "x4")
BASE_COVARIATE_COLS = ("x1", "x2", "x3", "x4", "region", "device")
SAMPLE_ESTIMANDS_WITH_COHORT_CENTERING = frozenset({"sample_ate"})


def world_noise_cols(spec: SyntheticWorldSpec) -> tuple[str, ...]:
    """Имена нерелевантных признаков ``noise_*`` мира."""

    count = max(spec.n_irrelevant_features, 24 if spec.high_dimensional else 0)
    return tuple(f"noise_{index:02d}" for index in range(count))


def world_pre_treatment_cols(spec: SyntheticWorldSpec) -> tuple[str, ...]:
    """Pre-treatment признаки, которые мир объявляет для анализа.

    По умолчанию совпадает с исторической схемой runner. Мир с
    ``noise_in_covariates=True`` добавляет все ``noise_*`` признаки, чтобы
    high-dimensional сценарий действительно передавал их в Lin/ML-RATE/CUPAC.
    """

    if spec.noise_in_covariates:
        return (*BASE_PRE_TREATMENT_COLS, *world_noise_cols(spec))
    return BASE_PRE_TREATMENT_COLS


def world_covariate_cols(spec: SyntheticWorldSpec) -> tuple[str, ...]:
    """Ковариаты ``ExperimentSpec.covariate_cols`` для мира."""

    if spec.noise_in_covariates:
        return (*BASE_COVARIATE_COLS, *world_noise_cols(spec))
    return BASE_COVARIATE_COLS


def has_heterogeneous_effect(spec: SyntheticWorldSpec) -> bool:
    """Есть ли у ``y1 - y0`` неконстантная часть."""

    return bool(spec.heterogeneous_effect or spec.treated_noise_scale != 1.0)


def spec_for_estimand(spec: SyntheticWorldSpec, estimand_id: str) -> SyntheticWorldSpec:
    """Вернуть спецификацию с центрированием, при котором H0 точно нулевая.

    Для ``sample_ate`` неоднородная часть эффекта центрируется по когорте
    (``effect=0`` даёт ``SATE == 0`` точно), для ``population_ate`` — по
    конечной популяции (``PATE == 0`` точно). Остальные estimand и миры без
    неоднородности возвращаются без изменений. Центрирование меняет только
    константу в ``y1``, поэтому оба варианта используют одинаковые случайные
    потоки и одну и ту же когорту.
    """

    if not has_heterogeneous_effect(spec):
        return spec
    if estimand_id in SAMPLE_ESTIMANDS_WITH_COHORT_CENTERING:
        centering = "cohort"
    elif estimand_id == "population_ate":
        centering = "population"
    else:
        return spec
    if spec.effect_modifier_centering == centering:
        return spec
    return replace(spec, effect_modifier_centering=centering)


def generate_world(
    spec: SyntheticWorldSpec | str,
    *,
    seed: int = 42,
) -> SyntheticWorld:
    """Сгенерировать мир до назначения treatment.

    ``assignment_data`` содержит только идентификатор, pre-treatment признаки и
    исторические значения. Потенциальные исходы и потенциальные B/C-признаки
    доступны отдельно для вычисления ground truth и последующей симуляции.

    ``spec.outcome_process="static"`` сохраняет исходный v3-процесс: outcome
    и pre-period панель порождены разными уравнениями. ``"panel"`` порождает
    единый ряд ``Y_it = alpha_i + f(X_i) + d_t + eps_it``: outcome — период
    ``T+1`` того же ряда, ``pre_target_lag1/lag2`` — периоды ``T`` и ``T-1``.
    Тогда ``pre_period_panel`` содержит только внешние сущности (вне когорты),
    и ``panel_shift``/``prior_mismatch`` искажают только эту внешнюю панель.

    ``spec.effect_modifier_centering`` определяет, какой estimand равен
    ``spec.effect`` точно: ``"population"`` (по умолчанию) центрирует
    неоднородную часть эффекта по конечной популяции, поэтому
    ``true_population_ate == effect``; ``"cohort"`` центрирует её по
    экспериментальной когорте, поэтому ``true_sample_ate == effect``. Истины
    обоих estimand для обоих вариантов центрирования записаны в ``metadata``.
    """

    if isinstance(spec, str):
        spec = get_world_spec(spec)
    rng = np.random.default_rng(seed)
    n = spec.population_size

    entity_id = np.arange(n, dtype=np.int64)
    latent = rng.normal(size=(n, 4))
    x1 = latent[:, 0]
    x2 = 0.45 * x1 + math.sqrt(1.0 - 0.45**2) * latent[:, 1]
    x3 = latent[:, 2]
    x4 = latent[:, 3]
    region_code = np.digitize(x2, (-0.8, -0.1, 0.6)).astype(np.int16)
    device_code = np.digitize(x3 + 0.25 * x1, (-0.7, 0.4)).astype(np.int16)

    signal = _prognostic_signal(spec, x1, x2, x3, x4)
    signal = _standardize(signal)
    panel_series: np.ndarray | None = None
    if spec.outcome_process == "panel":
        panel_series, persistent, noise = _panel_outcome_series(spec, rng, n, x1, signal)
        baseline = panel_series[:, -1]
        noise_scale = math.sqrt(1.0 - spec.r2 - spec.unit_effect_share)
        oracle_component = persistent
        x_component = math.sqrt(spec.r2) * signal
    else:
        noise = _outcome_noise(spec, rng, n, x1)
        baseline = math.sqrt(spec.r2) * signal + math.sqrt(1.0 - spec.r2) * noise
        noise_scale = math.sqrt(1.0 - spec.r2)
        oracle_component = math.sqrt(spec.r2) * signal
        x_component = oracle_component
        if spec.outcome_distribution == "zero_inflated_count":
            rate = np.exp(np.clip(0.6 + 0.45 * baseline, -2.0, 3.0))
            baseline = rng.poisson(rate).astype(float)
            baseline[rng.random(n) < 0.55] = 0.0
            oracle_component = None

    heterogeneity = _effect_heterogeneity(spec, x1, x3, oracle_component, noise, noise_scale, n)
    individual_effect = spec.effect + heterogeneity
    y0 = baseline
    y1 = baseline + individual_effect

    population = pd.DataFrame(
        {
            "entity_id": entity_id,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "region": pd.Categorical.from_codes(region_code, ["north", "south", "east", "west"]),
            "device": pd.Categorical.from_codes(device_code, ["mobile", "desktop", "tablet"]),
            "y0": y0,
            "y1": y1,
            "individual_effect": individual_effect,
        }
    )
    _add_irrelevant_features(population, spec, rng)
    _add_strata(population, spec, rng)
    _add_temporal_potentials(population, spec, rng)

    if panel_series is None:
        panel = _generate_pre_period_panel(population, spec, rng)
        cohort_index = _sample_cohort(population, spec, rng)
    else:
        cohort_index = _sample_cohort(population, spec, rng)
        panel = _external_panel(population, spec, panel_series, cohort_index)

    in_cohort = np.zeros(n, dtype=bool)
    in_cohort[cohort_index] = True
    heterogeneity_cohort_mean = float(np.mean(heterogeneity[cohort_index]))
    heterogeneity_population_mean = float(np.mean(heterogeneity))
    if spec.effect_modifier_centering == "cohort":
        _recenter_effect(population, spec, heterogeneity_cohort_mean)

    experiment = population.iloc[cohort_index].reset_index(drop=True)
    if panel_series is None:
        last_two = _latest_panel_features(panel, experiment["entity_id"])
    else:
        last_two = pd.DataFrame(
            {
                "entity_id": population["entity_id"].to_numpy()[cohort_index],
                "pre_target_lag1": panel_series[cohort_index, -2],
                "pre_target_lag2": panel_series[cohort_index, -3],
            }
        )
    experiment = experiment.merge(last_two, on="entity_id", how="left", validate="one_to_one")
    if spec.noise_in_covariates:
        for column in world_noise_cols(spec):
            # Static covariates have identical lag-2 values; temporal CUPAC A
            # needs the aligned ``__lag2`` copy of every declared feature.
            experiment[f"{column}__lag2"] = experiment[column]

    potential_columns = [
        "entity_id",
        "y0",
        "y1",
        "individual_effect",
        "zone_b0",
        "zone_b1",
        "zone_c0",
        "zone_c1",
        "post_outcome_proxy0",
        "post_outcome_proxy1",
    ]
    potential_outcomes = experiment[potential_columns].copy()
    assignment_columns = [
        column
        for column in experiment.columns
        if column not in _OUTCOME_COLUMNS and not column.startswith("post_")
    ]
    assignment_data = experiment[assignment_columns].copy()
    forbidden = _OUTCOME_COLUMNS.intersection(assignment_data.columns)
    if forbidden:
        raise RuntimeError(f"Outcome leakage into assignment_data: {sorted(forbidden)}")

    native_truths = _native_truths(population, experiment)
    true_sample_ate = float(np.mean(experiment["y1"] - experiment["y0"]))
    true_population_ate = float(np.mean(population["y1"] - population["y0"]))
    metadata = _world_metadata(
        spec,
        population,
        experiment,
        oracle_component=oracle_component,
        x_component=x_component,
        heterogeneity_cohort_mean=heterogeneity_cohort_mean,
        heterogeneity_population_mean=heterogeneity_population_mean,
        external_panel_entities=int(n - in_cohort.sum()) if panel_series is not None else n,
    )
    return SyntheticWorld(
        spec=spec,
        finite_population=population,
        pre_period_panel=panel,
        experiment_cohort=experiment,
        assignment_data=assignment_data,
        potential_outcomes=potential_outcomes,
        true_sample_ate=true_sample_ate,
        true_population_ate=true_population_ate,
        native_truths=MappingProxyType(native_truths),
        estimand_truths=MappingProxyType(
            {"sample_ate": true_sample_ate, "population_ate": true_population_ate}
        ),
        metadata=MappingProxyType(metadata),
    )


def _effect_heterogeneity(
    spec: SyntheticWorldSpec,
    x1: np.ndarray,
    x3: np.ndarray,
    oracle_component: np.ndarray | None,
    noise: np.ndarray,
    noise_scale: float,
    n: int,
) -> np.ndarray:
    """Non-constant part of ``y1 - y0``, centered over the finite population."""

    heterogeneity = np.zeros(n)
    if spec.heterogeneous_effect:
        if spec.effect_modifier_source == "prognostic":
            if oracle_component is None:
                raise ValueError("A prognostic effect modifier needs a continuous outcome.")
            source = oracle_component
        else:
            source = x1 + 0.5 * x3
        heterogeneity = spec.effect_modifier_scale * _standardize(source)
    if spec.treated_noise_scale != 1.0:
        # Arm-specific variance: the treated potential outcome carries the
        # same idiosyncratic shock with a different scale.
        extra = (spec.treated_noise_scale - 1.0) * noise_scale * np.asarray(noise, dtype=float)
        heterogeneity = heterogeneity + (extra - float(np.mean(extra)))
    return heterogeneity


def _recenter_effect(
    population: pd.DataFrame,
    spec: SyntheticWorldSpec,
    shift: float,
) -> None:
    """Move the effect heterogeneity so that the cohort mean equals ``effect``."""

    population["individual_effect"] = population["individual_effect"].to_numpy() - shift
    y1 = population["y1"].to_numpy() - shift
    population["y1"] = y1
    y0 = population["y0"].to_numpy()
    population["zone_b1"] = _potential_auxiliary(
        population["zone_b0"].to_numpy(), y0, y1, spec.zone_b_mode
    )
    population["zone_c1"] = _potential_auxiliary(
        population["zone_c0"].to_numpy(), y0, y1, spec.zone_c_mode
    )
    population["post_outcome_proxy1"] = population["post_outcome_proxy1"].to_numpy() - shift


def _panel_outcome_series(
    spec: SyntheticWorldSpec,
    rng: np.random.Generator,
    n: int,
    x1: np.ndarray,
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``Y`` of shape ``(n, T + 1)``, the persistent part and the last shock.

    Columns ``0..T-1`` are the pre-periods, column ``T`` is the experiment
    period. ``signal`` is already standardized, ``alpha`` is a persistent unit
    effect independent of the covariates, ``d_t`` is a common period shock and
    ``eps_it`` is an independent standardized shock with the spec's tail shape.
    """

    periods = spec.n_pre_periods + 1
    alpha = rng.normal(size=n)
    persistent = math.sqrt(spec.r2) * signal + math.sqrt(spec.unit_effect_share) * alpha
    shock_scale = math.sqrt(1.0 - spec.r2 - spec.unit_effect_share)
    period_effect = rng.normal(0.0, 0.2, periods)
    series = np.empty((n, periods), dtype=float)
    last_noise = np.zeros(n)
    for period in range(periods):
        last_noise = _outcome_noise(spec, rng, n, x1)
        series[:, period] = persistent + period_effect[period] + shock_scale * last_noise
    return series, persistent, last_noise


def _external_panel(
    population: pd.DataFrame,
    spec: SyntheticWorldSpec,
    series: np.ndarray,
    cohort_index: np.ndarray,
) -> pd.DataFrame:
    """Historical panel of entities outside the experiment cohort.

    Panel shift and prior mismatch are applied only here, so the aggregated
    panel prior can disagree with the experiment's own pre-period while the
    experiment's outcome and lags remain one stationary series.
    """

    outside = np.ones(len(population), dtype=bool)
    outside[cohort_index] = False
    rows = np.flatnonzero(outside)
    periods = spec.n_pre_periods
    months = pd.date_range("2023-01-01", periods=periods, freq="MS")
    x1 = population["x1"].to_numpy()[rows]
    x2 = population["x2"].to_numpy()[rows]
    distortion = spec.panel_shift * (0.6 * x1 + 0.3) + spec.prior_mismatch * (
        -1.1 * x1 + 0.9 * x2
    )
    values = series[rows, :periods] + distortion[:, None]
    count = rows.size
    return pd.DataFrame(
        {
            "entity_id": np.tile(population["entity_id"].to_numpy()[rows], periods),
            "report_date": np.repeat(months.to_numpy(), count),
            "month_index": np.repeat(np.arange(periods), count).astype(np.int16),
            "x1": np.tile(x1, periods),
            "x2": np.tile(x2, periods),
            "pre_target": values.T.reshape(-1),
        }
    )


def _r2_of_projection(target: np.ndarray, features: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    design = np.column_stack([np.ones(target.size), features])
    valid = np.isfinite(target) & np.isfinite(design).all(axis=1)
    if int(valid.sum()) <= design.shape[1]:
        return math.nan
    coefficients, *_ = np.linalg.lstsq(design[valid], target[valid], rcond=None)
    residual = target[valid] - design[valid] @ coefficients
    total = float(np.var(target[valid]))
    return float(1.0 - np.var(residual) / total) if total > 0.0 else math.nan


def _world_metadata(
    spec: SyntheticWorldSpec,
    population: pd.DataFrame,
    experiment: pd.DataFrame,
    *,
    oracle_component: np.ndarray | None,
    x_component: np.ndarray,
    heterogeneity_cohort_mean: float,
    heterogeneity_population_mean: float,
    external_panel_entities: int,
) -> dict[str, Any]:
    y0_population = population["y0"].to_numpy()
    y0_sample = experiment["y0"].to_numpy()
    pre = experiment["pre_target_lag1"].to_numpy(dtype=float)
    variance = float(np.var(y0_population))
    if oracle_component is None or variance <= 0.0:
        oracle_r2 = math.nan
        oracle_x_r2 = math.nan
    else:
        oracle_r2 = float(np.var(oracle_component) / variance)
        oracle_x_r2 = float(np.var(x_component) / variance)
    corr = np.corrcoef(y0_sample, pre)[0, 1] if np.std(pre) > 0.0 else math.nan
    covariates = experiment[list(world_pre_treatment_cols(spec))].to_numpy(dtype=float)
    # Truths under both centerings differ only by the constant shift of the
    # heterogeneous part, so they are exact without regenerating the world.
    effect = float(spec.effect)
    population_centered = {
        "sample_ate": effect + heterogeneity_cohort_mean,
        "population_ate": effect + heterogeneity_population_mean,
    }
    cohort_centered = {
        "sample_ate": effect + heterogeneity_cohort_mean - heterogeneity_cohort_mean,
        "population_ate": effect + heterogeneity_population_mean - heterogeneity_cohort_mean,
    }
    return {
        "outcome_process": spec.outcome_process,
        "effect_modifier_centering": spec.effect_modifier_centering,
        "oracle_r2": oracle_r2,
        "oracle_x_r2": oracle_x_r2,
        "oracle_corr2_y_pre": float(corr**2) if math.isfinite(corr) else math.nan,
        "linear_r2_pre_treatment_features": _r2_of_projection(y0_sample, covariates),
        "sample_minus_population_ate": heterogeneity_cohort_mean - heterogeneity_population_mean,
        "truths_population_centering": MappingProxyType(population_centered),
        "truths_cohort_centering": MappingProxyType(cohort_centered),
        "pre_treatment_cols": world_pre_treatment_cols(spec),
        "covariate_cols": world_covariate_cols(spec),
        "external_panel_entities": external_panel_entities,
    }


def _prognostic_signal(
    spec: SyntheticWorldSpec,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
) -> np.ndarray:
    if spec.signal_model == "linear":
        return 0.8 * x1 - 0.45 * x2 + 0.35 * x3
    if spec.signal_model == "quadratic":
        return 0.7 * x1 + 0.55 * (x2**2 - 1.0) - 0.3 * x3
    if spec.signal_model == "threshold":
        return 0.8 * (x1 > 0.2) - 0.65 * (x2 < -0.4) + 0.25 * x3
    return 0.65 * x1 * x2 - 0.5 * x3 * x4 + 0.35 * x1


def _outcome_noise(
    spec: SyntheticWorldSpec,
    rng: np.random.Generator,
    n: int,
    x1: np.ndarray,
) -> np.ndarray:
    if spec.outcome_distribution in {"gaussian", "zero_inflated_count"}:
        noise = rng.normal(size=n)
    elif spec.outcome_distribution == "student_t3":
        noise = rng.standard_t(3.0, size=n) / math.sqrt(3.0)
    elif spec.outcome_distribution == "lognormal":
        raw = rng.lognormal(0.0, 0.9, size=n)
        noise = _standardize(raw)
    else:
        contaminated = rng.random(n) < 0.03
        noise = rng.normal(0.0, np.where(contaminated, 10.0, 1.0), size=n)
        noise = _standardize(noise)
    if spec.heteroskedastic:
        noise = noise * (0.45 + 0.8 * np.abs(x1))
        noise = _standardize(noise)
    return np.asarray(noise, dtype=float)


def _generate_pre_period_panel(
    population: pd.DataFrame,
    spec: SyntheticWorldSpec,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(population)
    months = pd.date_range("2023-01-01", periods=spec.n_pre_periods, freq="MS")
    entity_id = np.tile(population["entity_id"].to_numpy(), spec.n_pre_periods)
    month_index = np.repeat(np.arange(spec.n_pre_periods), n)
    x1 = np.tile(population["x1"].to_numpy(), spec.n_pre_periods)
    x2 = np.tile(population["x2"].to_numpy(), spec.n_pre_periods)
    drift = spec.temporal_drift * month_index / max(spec.n_pre_periods - 1, 1)
    panel_signal = (0.75 + drift) * x1 - (0.35 - 0.5 * drift) * x2
    panel_signal += spec.panel_shift * (0.6 * x1 + 0.3)
    panel_signal += spec.prior_mismatch * (-1.1 * x1 + 0.9 * x2)
    month_shock = np.repeat(rng.normal(0.0, 0.2, spec.n_pre_periods), n)
    pre_target = panel_signal + month_shock + rng.normal(0.0, 0.65, n * spec.n_pre_periods)
    return pd.DataFrame(
        {
            "entity_id": entity_id,
            "report_date": np.repeat(months.to_numpy(), n),
            "month_index": month_index.astype(np.int16),
            "x1": x1,
            "x2": x2,
            "pre_target": pre_target,
        }
    )


def _latest_panel_features(panel: pd.DataFrame, entity_ids: pd.Series) -> pd.DataFrame:
    selected = panel[panel["entity_id"].isin(entity_ids)]
    wide = selected.pivot(index="entity_id", columns="month_index", values="pre_target")
    latest = wide.columns[-1]
    previous = wide.columns[-2]
    return pd.DataFrame(
        {
            "entity_id": wide.index.to_numpy(),
            "pre_target_lag1": wide[latest].to_numpy(),
            "pre_target_lag2": wide[previous].to_numpy(),
        }
    )


def _sample_cohort(
    population: pd.DataFrame,
    spec: SyntheticWorldSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    score = spec.cohort_shift * _standardize(population["x1"].to_numpy())
    score -= float(np.max(score))
    weights = np.exp(np.clip(score, -30.0, 0.0))
    weights /= weights.sum()
    return rng.choice(
        len(population),
        size=spec.n_experiment,
        replace=False,
        p=weights,
    )


def _add_irrelevant_features(
    population: pd.DataFrame,
    spec: SyntheticWorldSpec,
    rng: np.random.Generator,
) -> None:
    count = max(spec.n_irrelevant_features, 24 if spec.high_dimensional else 0)
    if count == 0:
        return
    values = rng.normal(size=(len(population), count)).astype(np.float32)
    for index in range(count):
        population[f"noise_{index:02d}"] = values[:, index]


def _add_strata(
    population: pd.DataFrame,
    spec: SyntheticWorldSpec,
    rng: np.random.Generator,
) -> None:
    if spec.many_rare_strata:
        category_a = np.minimum(rng.zipf(2.0, len(population)), 40)
        category_b = np.minimum(rng.zipf(2.3, len(population)), 25)
        population["strata_a"] = pd.Categorical(category_a.astype(str))
        population["strata_b"] = pd.Categorical(category_b.astype(str))
    else:
        population["strata_a"] = pd.Categorical(pd.qcut(population["x1"], 4, labels=False))
        population["strata_b"] = population["region"]


def _add_temporal_potentials(
    population: pd.DataFrame,
    spec: SyntheticWorldSpec,
    rng: np.random.Generator,
) -> None:
    base_b = 0.5 * population["x1"].to_numpy() + rng.normal(0.0, 0.5, len(population))
    base_c = 0.4 * population["x2"].to_numpy() + rng.normal(0.0, 0.5, len(population))
    y0 = population["y0"].to_numpy()
    y1 = population["y1"].to_numpy()

    population["zone_b0"] = base_b
    population["zone_b1"] = _potential_auxiliary(base_b, y0, y1, spec.zone_b_mode)
    population["zone_c0"] = base_c
    population["zone_c1"] = _potential_auxiliary(base_c, y0, y1, spec.zone_c_mode)
    proxy_noise = rng.normal(0.0, 0.05, len(population))
    population["post_outcome_proxy0"] = y0 + proxy_noise
    population["post_outcome_proxy1"] = y1 + proxy_noise


def _potential_auxiliary(
    base: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "safe":
        return base.copy()
    if mode == "treatment_dependent":
        return base + 0.5
    return base + 0.7 * (y1 - y0) + 0.25 * y1


def _native_truths(
    population: pd.DataFrame,
    sample: pd.DataFrame,
) -> dict[str, NativeEstimandTruth]:
    descriptions = {
        "trimmed_mean_10": "Разность 10% усечённых средних потенциальных исходов.",
        "winsorized_mean_10": "Разность 10% винзоризированных средних потенциальных исходов.",
        "median_effect": "Разность медиан потенциальных исходов.",
    }
    functions = {
        "trimmed_mean_10": lambda values: _trimmed_mean(values, 0.1),
        "winsorized_mean_10": lambda values: _winsorized_mean(values, 0.1),
        "median_effect": lambda values: float(np.median(values)),
    }
    result: dict[str, NativeEstimandTruth] = {}
    for estimand_id, function in functions.items():
        sample_value = function(sample["y1"].to_numpy()) - function(sample["y0"].to_numpy())
        population_value = function(population["y1"].to_numpy()) - function(population["y0"].to_numpy())
        result[estimand_id] = NativeEstimandTruth(
            estimand_id=estimand_id,
            sample_value=float(sample_value),
            population_value=float(population_value),
            description_ru=descriptions[estimand_id],
        )
    return result


def _trimmed_mean(values: np.ndarray, proportion: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    cut = int(math.floor(proportion * len(ordered)))
    return float(np.mean(ordered[cut : len(ordered) - cut]))


def _winsorized_mean(values: np.ndarray, proportion: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    cut = int(math.floor(proportion * len(ordered)))
    if cut == 0:
        return float(np.mean(ordered))
    clipped = np.clip(ordered, ordered[cut], ordered[-cut - 1])
    return float(np.mean(clipped))


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if std == 0.0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std
