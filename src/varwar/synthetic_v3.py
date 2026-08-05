from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

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


def generate_world(
    spec: SyntheticWorldSpec | str,
    *,
    seed: int = 42,
) -> SyntheticWorld:
    """Сгенерировать мир до назначения treatment.

    ``assignment_data`` содержит только идентификатор, pre-treatment признаки и
    исторические значения. Потенциальные исходы и потенциальные B/C-признаки
    доступны отдельно для вычисления ground truth и последующей симуляции.
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
    noise = _outcome_noise(spec, rng, n, x1)
    baseline = math.sqrt(spec.r2) * signal + math.sqrt(1.0 - spec.r2) * noise
    if spec.outcome_distribution == "zero_inflated_count":
        rate = np.exp(np.clip(0.6 + 0.45 * baseline, -2.0, 3.0))
        baseline = rng.poisson(rate).astype(float)
        baseline[rng.random(n) < 0.55] = 0.0

    effect_modifier = 0.35 * _standardize(x1 + 0.5 * x3)
    individual_effect = spec.effect + (
        effect_modifier if spec.heterogeneous_effect else np.zeros(n)
    )
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

    panel = _generate_pre_period_panel(population, spec, rng)
    cohort_index = _sample_cohort(population, spec, rng)
    experiment = population.iloc[cohort_index].reset_index(drop=True)
    last_two = _latest_panel_features(panel, experiment["entity_id"])
    experiment = experiment.merge(last_two, on="entity_id", how="left", validate="one_to_one")

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
    )


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
