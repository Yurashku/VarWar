from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

from varwar.core import ExperimentSpec, WrappedEstimator
from varwar.estimators import (
    BootstrapMeanDiffEstimator,
    DifferenceInMeansEstimator,
    HypExABTestEstimator,
    HypExCupacEstimator,
    HypExCupedEstimator,
    HypExMatchingEstimator,
    MLRateEstimator,
    NearestNeighborMatchingEstimator,
    OptimalTransportMatchingEstimator,
    PermutationMeanDiffEstimator,
    RatioDeltaMethodEstimator,
    RandomizedWeightedMeanDiffEstimator,
    RegressionAdjustmentEstimator,
    StratifiedDifferenceInMeansEstimator,
    TrimmedMeanEstimator,
    VarianceWeightedStratifiedEstimator,
    WaldEstimator,
    WinsorizedMeanEstimator,
)
from varwar.feature_sets import CupacFeaturePolicy, default_cupac_feature_sets
from varwar.panel_prior import PanelPriorSummary
from varwar.transforms import (
    CupedTransformer,
    QuadraticCupacTransformer,
    WinsorizeTransformer,
    ZonedCupacTransformer,
    PanelPriorZonedCupacTransformer,
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
    design_mode: str = "randomized"
    information_mode: str = "experiment_only"
    estimand_id: str = "sample_ate"
    temporal_mode: str = "pre_treatment_only"

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
            "design_mode": self.design_mode,
            "information_mode": self.information_mode,
            "estimand_id": self.estimand_id,
            "temporal_mode": self.temporal_mode,
        }


@dataclass
class PipelineCandidate:
    definition: PipelineDefinition
    estimator: Any


def _with_temporal_modes(candidates: list[PipelineCandidate]) -> list[PipelineCandidate]:
    """Attach an explicit future-information contract to every pipeline."""

    result: list[PipelineCandidate] = []
    for candidate in candidates:
        pipeline_id = candidate.definition.pipeline_id
        post_treatment = any(
            marker in pipeline_id
            for marker in ("plus_B", "plus_C", "_B__", "_C__")
        )
        result.append(
            PipelineCandidate(
                definition=replace(
                    candidate.definition,
                    temporal_mode=(
                        "post_treatment_allowed"
                        if post_treatment
                        else "pre_treatment_only"
                    ),
                ),
                estimator=candidate.estimator,
            )
        )
    return result


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
    include_specialized: bool = True,
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
                "pooled_mean_diff",
                "Разность средних с общей дисперсией",
                ("pooled_mean_diff",),
                "ATE",
                PipelineTier.EXPLORATORY,
                "Текущие treatment и target; диагностический legacy-контур.",
                "Требует допущения равенства дисперсий.",
                "welch",
            ),
            DifferenceInMeansEstimator(name="pooled_mean_diff", variance="pooled"),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "randomized_weighted_mean_diff",
                "Воспроизводимая взвешенная разность средних",
                ("randomized_weighted_mean_diff",),
                "Взвешенный ATE",
                PipelineTier.EXPLORATORY,
                "Текущие treatment и target; фиксированный seed.",
                "Цель задаётся механизмом весов.",
                "welch",
            ),
            RandomizedWeightedMeanDiffEstimator(),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "permutation_mean_diff",
                "Перестановочная оценка разности средних",
                ("permutation_mean_diff",),
                "ATE",
                PipelineTier.EXPLORATORY,
                "Только текущие treatment и target.",
                "Permutation p-value и interval через инверсию permutation-теста.",
                "welch",
            ),
            PermutationMeanDiffEstimator(n_resamples=499),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "bootstrap_mean_diff",
                "Bootstrap-оценка разности средних",
                ("bootstrap_mean_diff",),
                "ATE",
                PipelineTier.EXPLORATORY,
                "Только текущие treatment и target.",
                "Согласованные studentized bootstrap-t интервал и p-value.",
                "welch",
            ),
            BootstrapMeanDiffEstimator(n_resamples=499),
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
                    "nearest_neighbor_matching",
                    "Matching ближайшего соседа",
                    ("nearest_neighbor_matching",),
                    "ATE",
                    PipelineTier.EXPLORATORY,
                    "Текущий target и числовые pre-treatment ковариаты.",
                    "Не сравнивать напрямую с ATE-пайплайнами.",
                    None,
                ),
                NearestNeighborMatchingEstimator(estimand="ate"),
            ),
            PipelineCandidate(
                PipelineDefinition(
                    "hypex_matching",
                    "Matching HypEx",
                    ("hypex_matching",),
                    "ATE",
                    PipelineTier.INTEGRATION_PARITY,
                    "Текущий target и pre-treatment ковариаты; нужен PyPI HypEx.",
                    "Двусторонний matching ATT/ATC с bias correction внутри HypEx.",
                    None,
                ),
                HypExMatchingEstimator(),
            ),
            PipelineCandidate(
                PipelineDefinition(
                    "optimal_transport_matching",
                    "Глобальный matching оптимальным транспортом",
                    ("optimal_transport_matching",),
                    "ATE",
                    PipelineTier.EXPLORATORY,
                    "Текущий target и числовые pre-treatment ковариаты.",
                    "Точный квадратичный Monge plan; минимальная capacity при дефиците control.",
                    None,
                ),
                OptimalTransportMatchingEstimator(estimand="ate"),
            ),
            PipelineCandidate(
                PipelineDefinition(
                    "trimmed_mean_diff",
                    "Разность усечённых средних",
                    ("trimmed_mean_diff",),
                    "Усечённая разность средних",
                    PipelineTier.CHANGED_METRIC,
                    "Только текущие treatment и target.",
                    "Изменяет target и состав наблюдений.",
                    None,
                ),
                TrimmedMeanEstimator(),
            ),
            PipelineCandidate(
                PipelineDefinition(
                    "winsorized_mean_diff",
                    "Разность винзоризированных средних",
                    ("winsorized_mean_diff",),
                    "Винзоризированная разность средних",
                    PipelineTier.CHANGED_METRIC,
                    "Только текущие treatment и target.",
                    "Историческая форма; в новых пайплайнах предпочтителен transformer.",
                    None,
                ),
                WinsorizedMeanEstimator(),
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
    if include_specialized and spec.numerator_col and spec.denominator_col:
        candidates.append(
            PipelineCandidate(
                PipelineDefinition(
                    "ratio_delta",
                    "Delta-оценка ratio-метрики",
                    ("ratio_delta",),
                    "Разность отношений средних",
                    PipelineTier.CHANGED_METRIC,
                    "Нужны numerator_col и denominator_col.",
                    "Сравнивается только с ratio-пайплайнами.",
                    None,
                ),
                RatioDeltaMethodEstimator(),
            )
        )
    if include_specialized and spec.instrument_col and spec.received_treatment_col:
        candidates.append(
            PipelineCandidate(
                PipelineDefinition(
                    "wald_late",
                    "Оценка Вальда для LATE",
                    ("wald_late",),
                    "LATE",
                    PipelineTier.OBSERVATIONAL,
                    "Нужны instrument_col и received_treatment_col.",
                    "Сравнивается только с IV-пайплайнами.",
                    None,
                ),
                WaldEstimator(),
            )
        )
    return _with_temporal_modes(candidates)


def build_stratified_pipelines(
    spec: ExperimentSpec,
    stratum_weights: Mapping[Any, float] | None = None,
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

    estimand = (
        "ATE экспериментальной выборки"
        if stratum_weights is None
        else "ATE с заранее заданными внешними весами страт"
    )
    weight_contract = (
        "Веса макрострат оцениваются по составу текущего A/B-датасета."
        if stratum_weights is None
        else "Веса макрострат переданы извне и должны быть зафиксированы до анализа."
    )

    candidates = [
        PipelineCandidate(
            PipelineDefinition(
                "ga_strata__stratified",
                "GA-страты -> стратифицированная оценка",
                ("genetic_strata_reducer", "stratified_mean_diff"),
                estimand,
                PipelineTier.CONFIRMATORY,
                "GA обучается на pre-target тех же строк без текущего outcome.",
                weight_contract,
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
                estimand,
                PipelineTier.CONFIRMATORY,
                "GA использует pre-target; CUPED theta оценивается по контролю.",
                weight_contract,
                "ga_strata__stratified",
            ),
            WrappedEstimator(
                CupedTransformer(fit_data="experiment_control"),
                base(),
                name="ga_strata__cuped_control__stratified",
            ),
        ),
        PipelineCandidate(
            PipelineDefinition(
                "variance_weighted_stratified",
                "Обратное дисперсионное взвешивание страт",
                ("variance_weighted_stratified",),
                "Precision-weighted effect",
                PipelineTier.EXPLORATORY,
                "Нужна колонка stratum_col; малые страты пропускаются.",
                "Не является общим ATE при неоднородном эффекте.",
                "ga_strata__stratified",
            ),
            VarianceWeightedStratifiedEstimator(),
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
                    estimand,
                    tier,
                    (
                        "GA и CUPAC A используют pre-period данные; B/C second-stage "
                        "оцениваются по текущему контролю."
                    ),
                    weight_contract,
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
    return _with_temporal_modes(candidates)


def build_panel_prior_pipelines(
    spec: ExperimentSpec,
    summary: PanelPriorSummary | None,
    *,
    prefer_pypi_hypex: bool = True,
) -> list[PipelineCandidate]:
    """Build A, A+B and A+B+C CUPAC pipelines using a panel prior summary."""

    policies = _policy_by_id(spec)
    candidates: list[PipelineCandidate] = []
    for policy in default_cupac_feature_sets(spec):
        panel_method_id = policy.method_id.replace("cupac_linear", "panel_prior_cupac")
        pipeline_id = f"{panel_method_id}__hypex_ab"
        candidates.append(
            PipelineCandidate(
                PipelineDefinition(
                    pipeline_id,
                    f"Panel-prior {policy.method_id} -> HypEx A/B",
                    (panel_method_id, "hypex_ab"),
                    "ATE",
                    PipelineTier.CONFIRMATORY if policy.zones == ("A",) else PipelineTier.EXPLORATORY,
                    "A/B target, pre-period zones and aggregated historical panel sufficient statistics.",
                    "Panel prior is discounted by temporal validation; B/C remain separate stages.",
                    (
                        policy.predecessor.replace("cupac_linear", "panel_prior_cupac")
                        if policy.predecessor
                        else "welch"
                    ),
                    information_mode="panel_prior",
                ),
                WrappedEstimator(
                    PanelPriorZonedCupacTransformer(policy=policy, summary=summary),
                    _hypex(prefer_pypi_hypex),
                    name=pipeline_id,
                ),
            )
        )
    return _with_temporal_modes(candidates)


def build_cartesian_rct_pipelines(
    spec: ExperimentSpec,
    *,
    prefer_pypi_hypex: bool = True,
    include_in_time: bool = True,
    include_panel_prior: bool = False,
    panel_summary: PanelPriorSummary | None = None,
) -> list[PipelineCandidate]:
    """Построить декартово произведение совместимых RCT-компонентов.

    В произведение входят базовые оценщики среднего эффекта и target-wrapper-ы.
    Генетическое укрупнение страт намеренно остаётся отдельным препроцессором:
    ему требуется изменить `ExperimentSpec.stratum_col` и подготовить данные до
    вызова оценщика, поэтому его комбинации запускаются отдельной гонкой GA.

    ID имеют форму ``cartesian__<estimator>__<transformer>``. Например,
    ``cartesian__welch__cuped_control`` и
    ``cartesian__hypex_ab__cupac_linear_A_plus_B``. Если panel-prior включён,
    соответствующие строки получают `information_mode="panel_prior"`.
    """

    policies = _policy_by_id(spec)
    base_factories: dict[str, Any] = {
        "welch": lambda: DifferenceInMeansEstimator(name="welch"),
        "hypex_ab": lambda: _hypex(prefer_pypi_hypex),
        "permutation": lambda: PermutationMeanDiffEstimator(),
        "bootstrap": lambda: BootstrapMeanDiffEstimator(),
        "regression_adjustment": lambda: RegressionAdjustmentEstimator(),
        "ml_rate": lambda: MLRateEstimator(),
        "nearest_neighbor_matching": lambda: NearestNeighborMatchingEstimator(
            estimand="ate"
        ),
        "optimal_transport_matching": lambda: OptimalTransportMatchingEstimator(
            estimand="ate"
        ),
    }
    transformer_ids = [
        "raw",
        "cuped_control",
        "cupac_linear_A",
        *( ["cupac_linear_A_plus_B", "cupac_linear_A_plus_B_plus_C"] if include_in_time else [] ),
        "cupac_quadratic_A",
    ]
    if include_panel_prior:
        transformer_ids.extend(
            [
                "panel_prior_cupac_A",
                *(
                    [
                        "panel_prior_cupac_A_plus_B",
                        "panel_prior_cupac_A_plus_B_plus_C",
                    ]
                    if include_in_time
                    else []
                ),
            ]
        )

    def wrapped(transformer_id: str, base: Any, name: str) -> Any:
        if transformer_id == "raw":
            return base
        if transformer_id == "cuped_control":
            return WrappedEstimator(
                CupedTransformer(fit_data="experiment_control"),
                base,
                name=name,
            )
        if transformer_id == "cupac_quadratic_A":
            return WrappedEstimator(
                QuadraticCupacTransformer(
                    feature_cols=tuple(spec.pre_treatment_features())
                ),
                base,
                name=name,
            )
        if transformer_id.startswith("panel_prior_cupac_"):
            policy_id = transformer_id.replace("panel_prior_cupac", "cupac_linear")
            return WrappedEstimator(
                PanelPriorZonedCupacTransformer(
                    policy=policies[policy_id],
                    summary=panel_summary,
                ),
                base,
                name=name,
            )
        return WrappedEstimator(
            ZonedCupacTransformer(policy=policies[transformer_id]),
            base,
            name=name,
        )

    candidates: list[PipelineCandidate] = []
    for estimator_id, factory in base_factories.items():
        for transformer_id in transformer_ids:
            pipeline_id = f"cartesian__{estimator_id}__{transformer_id}"
            temporal_mode = (
                "post_treatment_allowed"
                if ("plus_B" in transformer_id or "plus_C" in transformer_id)
                else "pre_treatment_only"
            )
            info_mode = (
                "panel_prior"
                if transformer_id.startswith("panel_prior_")
                else "experiment_only"
            )
            label = f"{estimator_id} + {transformer_id}"
            candidates.append(
                PipelineCandidate(
                    PipelineDefinition(
                        pipeline_id,
                        label,
                        (transformer_id, estimator_id),
                        "ATE",
                        PipelineTier.EXPLORATORY,
                        "Randomized experiment, compatible target wrapper and base estimator.",
                        "Cartesian component product; inspect each temporal and information contract separately.",
                        None,
                        information_mode=info_mode,
                        temporal_mode=temporal_mode,
                    ),
                    wrapped(transformer_id, factory(), pipeline_id),
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
