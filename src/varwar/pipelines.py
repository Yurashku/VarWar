from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from varwar.core import ExperimentSpec, WrappedEstimator
from varwar.estimators import (
    DifferenceInMeansEstimator,
    HypExABTestEstimator,
    HypExCupacEstimator,
    HypExCupedEstimator,
    MLRateEstimator,
    RegressionAdjustmentEstimator,
    StratifiedDifferenceInMeansEstimator,
)
from varwar.feature_sets import CupacFeaturePolicy, default_cupac_feature_sets
from varwar.transforms import (
    CupedTransformer,
    QuadraticCupacTransformer,
    WinsorizeTransformer,
    ZonedCupacTransformer,
)


class PipelineTier(str, Enum):
    """Роль конвейера в R&D-гонке."""

    CONFIRMATORY = "confirmatory"
    INTEGRATION_PARITY = "integration_parity"
    EXPLORATORY = "exploratory"
    CHANGED_METRIC = "changed_metric"
    OBSERVATIONAL = "observational"


@dataclass(frozen=True)
class PipelineDefinition:
    """Стабильное описание полного анализа, независимое от его реализации."""

    pipeline_id: str
    name_ru: str
    components: tuple[str, ...]
    estimand: str
    tier: PipelineTier
    data_contract: str
    notes: str = ""
    predecessor_pipeline: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline_id,
            "pipeline_name_ru": self.name_ru,
            "components": " -> ".join(self.components),
            "estimand": self.estimand,
            "pipeline_tier": self.tier.value,
            "data_contract": self.data_contract,
            "pipeline_notes": self.notes,
            "predecessor_pipeline": self.predecessor_pipeline,
        }


@dataclass
class PipelineCandidate:
    definition: PipelineDefinition
    estimator: Any


def _hypex(prefer_pypi: bool) -> HypExABTestEstimator:
    return HypExABTestEstimator(
        prefer_pypi=prefer_pypi,
        require_pypi=prefer_pypi,
    )


def _policy_by_id(spec: ExperimentSpec) -> Mapping[str, CupacFeaturePolicy]:
    return {
        policy.method_id: policy
        for policy in default_cupac_feature_sets(spec)
    }


def build_full_traffic_pipelines(
    spec: ExperimentSpec,
    *,
    prefer_pypi_hypex: bool = True,
) -> list[PipelineCandidate]:
    """Собрать допустимые full-traffic пайплайны без декартового взрыва.

    В список входят статистический baseline, интеграционные проверки HypEx,
    основные корректировки target, ковариатные estimators и один отдельный
    контур для изменённой винзоризацией метрики.
    """

    policies = _policy_by_id(spec)
    a = spec.pre_treatment_features()
    candidates = [
        PipelineCandidate(
            PipelineDefinition(
                "welch",
                "Welch t-test",
                ("welch_mean_diff",),
                "ATE",
                PipelineTier.CONFIRMATORY,
                "Только текущие treatment и target.",
            ),
            DifferenceInMeansEstimator(name="welch"),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "hypex_ab",
                "HypEx A/B (Welch)",
                ("hypex_ab",),
                "ATE",
                PipelineTier.INTEGRATION_PARITY,
                "Только текущие treatment и target.",
                "Должен совпадать с локальным Welch с точностью вычислений.",
                "welch",
            ),
            _hypex(prefer_pypi_hypex),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "hypex_cuped",
                "HypEx CUPED",
                ("hypex_cuped", "hypex_ab"),
                "ATE",
                PipelineTier.INTEGRATION_PARITY,
                "Theta оценивается HypEx на всём текущем эксперименте.",
                "",
                "hypex_ab",
            ),
            HypExCupedEstimator(
                prefer_pypi=prefer_pypi_hypex,
                require_pypi=prefer_pypi_hypex,
            ),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "cuped_control__hypex_ab",
                "CUPED по контролю -> HypEx A/B",
                ("cuped_control", "hypex_ab"),
                "ATE",
                PipelineTier.CONFIRMATORY,
                "Theta оценивается по target и pre-target текущего контроля.",
                "",
                "hypex_ab",
            ),
            WrappedEstimator(
                CupedTransformer(fit_data="experiment_control"),
                _hypex(prefer_pypi_hypex),
                name="cuped_control__hypex_ab",
            ),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "hypex_cupac_temporal",
                "HypEx temporal CUPAC",
                ("hypex_cupac_temporal", "hypex_ab"),
                "ATE",
                PipelineTier.INTEGRATION_PARITY,
                "Модель lag2 -> lag1; текущий outcome не используется для обучения.",
                "",
                "hypex_ab",
            ),
            HypExCupacEstimator(
                cupac_features=a,
                prefer_pypi=prefer_pypi_hypex,
                require_pypi=prefer_pypi_hypex,
            ),
        ),
    ]

    for method_id, label, tier in (
        ("cupac_linear_A", "CUPAC A -> HypEx A/B", PipelineTier.CONFIRMATORY),
        (
            "cupac_linear_A_plus_B",
            "CUPAC A+B -> HypEx A/B",
            PipelineTier.EXPLORATORY,
        ),
        (
            "cupac_linear_A_plus_B_plus_C",
            "CUPAC A+B+C -> HypEx A/B",
            PipelineTier.EXPLORATORY,
        ),
    ):
        policy = policies[method_id]
        predecessor = {
            "hypex_ab": "hypex_ab",
            "cupac_linear_A": "cupac_linear_A__hypex_ab",
            "cupac_linear_A_plus_B": "cupac_linear_A_plus_B__hypex_ab",
        }.get(policy.predecessor or "")
        candidates.append(
            PipelineCandidate(
                PipelineDefinition(
                    f"{method_id}__hypex_ab",
                    label,
                    (method_id, "hypex_ab"),
                    "ATE",
                    tier,
                    (
                        "Temporal CUPAC A (lag2 -> lag1) на том же датасете; "
                        "B и C выполняются отдельными линейными этапами."
                    ),
                    (
                        "Баланс зоны C является диагностическим фильтром, "
                        "а не доказательством причинной безопасности."
                        if "C" in policy.zones
                        else ""
                    ),
                    predecessor,
                ),
                WrappedEstimator(
                    ZonedCupacTransformer(policy=policy),
                    _hypex(prefer_pypi_hypex),
                    name=f"{method_id}__hypex_ab",
                ),
            )
        )

    candidates.extend(
        [
            PipelineCandidate(
                PipelineDefinition(
                    "cupac_quadratic_A__hypex_ab",
                    "Квадратичный CUPAC A -> HypEx A/B",
                    ("cupac_quadratic_A", "hypex_ab"),
                    "ATE",
                    PipelineTier.EXPLORATORY,
                    "Temporal lag2 -> lag1; только pre-experiment зона A.",
                    "Число модельных колонок растёт квадратично с числом признаков.",
                    "cupac_linear_A__hypex_ab",
                ),
                WrappedEstimator(
                    QuadraticCupacTransformer(feature_cols=a),
                    _hypex(prefer_pypi_hypex),
                    name="cupac_quadratic_A__hypex_ab",
                ),
            ),
            PipelineCandidate(
                PipelineDefinition(
                    "regression_adjustment",
                    "Регрессионная корректировка",
                    ("regression_adjustment",),
                    "ATE",
                    PipelineTier.CONFIRMATORY,
                    "Текущий target и только pre-experiment ковариаты.",
                    "",
                    "welch",
                ),
                RegressionAdjustmentEstimator(),
            ),
            PipelineCandidate(
                PipelineDefinition(
                    "ml_rate",
                    "ML-RATE",
                    ("ml_rate",),
                    "ATE",
                    PipelineTier.EXPLORATORY,
                    "Cross-fitting по текущему датасету; pre-experiment признаки.",
                    "",
                    "welch",
                ),
                MLRateEstimator(),
            ),
            PipelineCandidate(
                PipelineDefinition(
                    "winsorized__hypex_ab",
                    "Винзоризация -> HypEx A/B",
                    ("winsorized", "hypex_ab"),
                    "ATE винзоризированного target",
                    PipelineTier.CHANGED_METRIC,
                    "Пороги оцениваются по текущему контролю.",
                    "Нельзя ранжировать по смещению относительно исходного ATE.",
                    None,
                ),
                WrappedEstimator(
                    WinsorizeTransformer(fit_data="experiment_control"),
                    _hypex(prefer_pypi_hypex),
                    name="winsorized__hypex_ab",
                ),
            ),
        ]
    )
    return candidates


def build_stratified_pipelines(
    spec: ExperimentSpec,
    stratum_weights: Mapping[Any, float],
    *,
    prefer_pypi_hypex: bool = True,
) -> list[PipelineCandidate]:
    """Собрать ограниченный набор GA -> stratified ablation-пайплайнов."""

    del prefer_pypi_hypex
    policies = _policy_by_id(spec)

    def base() -> StratifiedDifferenceInMeansEstimator:
        return StratifiedDifferenceInMeansEstimator(
            stratum_weights=stratum_weights,
        )

    candidates = [
        PipelineCandidate(
            PipelineDefinition(
                "ga_strata__stratified",
                "GA-страты -> стратифицированная оценка",
                ("genetic_strata_reducer", "stratified_mean_diff"),
                "ATE с фиксированными pre-period весами страт",
                PipelineTier.CONFIRMATORY,
                "GA обучается на pre-target тех же строк без текущего outcome.",
                "",
                "welch",
            ),
            base(),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "ga_strata__cuped_control__stratified",
                "GA-страты -> CUPED -> стратифицированная оценка",
                (
                    "genetic_strata_reducer",
                    "cuped_control",
                    "stratified_mean_diff",
                ),
                "ATE с фиксированными pre-period весами страт",
                PipelineTier.CONFIRMATORY,
                "GA использует pre-target; CUPED theta оценивается по контролю.",
                "",
                "ga_strata__stratified",
            ),
            WrappedEstimator(
                CupedTransformer(fit_data="experiment_control"),
                base(),
                name="ga_strata__cuped_control__stratified",
            ),
        ),
    ]
    for method_id, label, tier in (
        (
            "cupac_linear_A",
            "GA-страты -> CUPAC A -> стратифицированная оценка",
            PipelineTier.CONFIRMATORY,
        ),
        (
            "cupac_linear_A_plus_B",
            "GA-страты -> CUPAC A+B -> стратифицированная оценка",
            PipelineTier.EXPLORATORY,
        ),
        (
            "cupac_linear_A_plus_B_plus_C",
            "GA-страты -> CUPAC A+B+C -> стратифицированная оценка",
            PipelineTier.EXPLORATORY,
        ),
    ):
        candidates.append(
            PipelineCandidate(
                PipelineDefinition(
                    f"ga_strata__{method_id}__stratified",
                    label,
                    (
                        "genetic_strata_reducer",
                        method_id,
                        "stratified_mean_diff",
                    ),
                    "ATE с фиксированными pre-period весами страт",
                    tier,
                    (
                        "GA и CUPAC A используют pre-period данные; B/C second-stage "
                        "оцениваются по текущему контролю."
                    ),
                    "",
                    (
                        "ga_strata__stratified"
                        if method_id == "cupac_linear_A"
                        else (
                            "ga_strata__cupac_linear_A__stratified"
                            if method_id == "cupac_linear_A_plus_B"
                            else "ga_strata__cupac_linear_A_plus_B__stratified"
                        )
                    ),
                ),
                WrappedEstimator(
                    ZonedCupacTransformer(policy=policies[method_id]),
                    base(),
                    name=f"ga_strata__{method_id}__stratified",
                ),
            )
        )
    return candidates


def pipeline_catalog(
    spec: ExperimentSpec,
    *,
    prefer_pypi_hypex: bool = True,
) -> list[PipelineDefinition]:
    """Вернуть каталог full-traffic пайплайнов без их выполнения."""

    return [
        candidate.definition
        for candidate in build_full_traffic_pipelines(
            spec,
            prefer_pypi_hypex=prefer_pypi_hypex,
        )
    ]
