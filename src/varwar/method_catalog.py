from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MethodCategory(str, Enum):
    """Роль публичного метода в экспериментальном конвейере."""

    ESTIMATOR = "Estimator"
    HYPOTHESIS_TEST = "HypothesisTest"
    TARGET_TRANSFORMER = "TargetTransformer"
    METRIC_TRANSFORMER = "MetricTransformer"
    DESIGN_PREPROCESSOR = "DesignPreprocessor"
    FEATURE_ADAPTER = "FeatureAdapter"
    INFERENCE_CORRECTION = "InferenceCorrection"
    CONFIGURATION_PROFILE = "ConfigurationProfile"
    OPTIMIZATION_OBJECTIVE = "OptimizationObjective"
    COMPOSITE_PIPELINE = "CompositePipeline"


class MethodStatus(str, Enum):
    """Зрелость реализации без заявления о production-ready качестве."""

    IMPLEMENTED = "implemented"
    IMPLEMENTED_INTEGRATION = "implemented_integration"
    IMPLEMENTED_LEGACY = "implemented_legacy"
    PLACEHOLDER_RND = "placeholder_rnd"
    PLACEHOLDER_NEEDS_SPEC = "placeholder_needs_spec"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class MethodRecord:
    """Поисковые метаданные компонента, профиля или участника гонки."""

    method_id: str
    category: MethodCategory
    status: MethodStatus
    summary: str
    implementation: str
    race_scope: str = "component"

    @property
    def is_implemented(self) -> bool:
        return self.status in {
            MethodStatus.IMPLEMENTED,
            MethodStatus.IMPLEMENTED_INTEGRATION,
            MethodStatus.IMPLEMENTED_LEGACY,
        }


_METHODS = (
    MethodRecord(
        "welch_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference in means with Welch inference; canonical unadjusted A/B baseline.",
        "varwar.estimators.DifferenceInMeansEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "hypex_ab",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "PyPI HypEx A/B integration baseline; its default t-test uses Welch unequal-variance inference.",
        "varwar.estimators.HypExABTestEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "pooled_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED_LEGACY,
        "The same mean difference with Student pooled-variance inference; retained only for diagnostics.",
        "varwar.estimators.DifferenceInMeansEstimator",
        "excluded_from_primary_race",
    ),
    MethodRecord(
        "randomized_weighted_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference of normalized hedgehog-weighted means with seeded within-arm shuffling and coherent Welch-style inference.",
        "varwar.estimators.RandomizedWeightedMeanDiffEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "permutation_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference-in-means estimator with permutation-calibrated p-value and inverted permutation confidence interval.",
        "varwar.estimators.PermutationMeanDiffEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "bootstrap_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference-in-means estimator with within-arm bootstrap standard error and percentile interval.",
        "varwar.estimators.BootstrapMeanDiffEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "regression_adjustment",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "OLS treatment coefficient adjusted for pre-treatment covariates with HC1 inference.",
        "varwar.estimators.RegressionAdjustmentEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "ml_rate",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Cross-fitted prognostic ridge predictions followed by interacted robust regression adjustment.",
        "varwar.estimators.MLRateEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "stratified_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Stratum-weighted average of within-stratum treatment effects for the same weighted ATE.",
        "varwar.estimators.StratifiedDifferenceInMeansEstimator",
        "stratification_pipeline_race",
    ),
    MethodRecord(
        "variance_weighted_stratified",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Inverse-variance combination of stratum effects; may target a precision-weighted effect instead of ATE.",
        "varwar.estimators.VarianceWeightedStratifiedEstimator",
        "specialized_only",
    ),
    MethodRecord(
        "nearest_neighbor_matching",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "One-nearest-neighbour ATT estimator with standardized numeric covariates and replacement.",
        "varwar.estimators.NearestNeighborMatchingEstimator",
        "observational_specialized",
    ),
    MethodRecord(
        "ratio_delta",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference of ratio-of-means metrics with direct delta-method inference.",
        "varwar.estimators.RatioDeltaMethodEstimator",
        "ratio_metric_specialized",
    ),
    MethodRecord(
        "wald_late",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Wald ratio estimator of LATE under a valid instrument and non-zero first stage.",
        "varwar.estimators.WaldEstimator",
        "instrumental_variable_specialized",
    ),
    MethodRecord(
        "trimmed_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED_LEGACY,
        "Legacy difference of group-wise trimmed means; changes the estimand and is not a primary ATE competitor.",
        "varwar.estimators.TrimmedMeanEstimator",
        "excluded_from_primary_race",
    ),
    MethodRecord(
        "winsorized_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED_LEGACY,
        "Legacy difference of group-wise winsorized means; use the composable winsorization transformer instead.",
        "varwar.estimators.WinsorizedMeanEstimator",
        "excluded_from_primary_race",
    ),
    MethodRecord(
        "ks_test",
        MethodCategory.HYPOTHESIS_TEST,
        MethodStatus.IMPLEMENTED,
        "Two-sample Kolmogorov-Smirnov test for any distributional difference.",
        "varwar.estimators.KSTestEstimator",
        "distribution_test_benchmark",
    ),
    MethodRecord(
        "approx_ks_fast",
        MethodCategory.HYPOTHESIS_TEST,
        MethodStatus.IMPLEMENTED,
        "Histogram approximation of the two-sample KS statistic with linear memory-bounded evaluation.",
        "varwar.estimators.ApproxKSTestEstimator",
        "distribution_test_benchmark",
    ),
    MethodRecord(
        "energy_test",
        MethodCategory.HYPOTHESIS_TEST,
        MethodStatus.IMPLEMENTED,
        "Energy-distance two-sample test with bounded-memory permutations.",
        "varwar.estimators.EnergyDistanceTestEstimator",
        "distribution_test_benchmark",
    ),
    MethodRecord(
        "randomized_t_test",
        MethodCategory.HYPOTHESIS_TEST,
        MethodStatus.IMPLEMENTED,
        "Randomized hedgehog-weighted t statistic for small-sample research.",
        "varwar.estimators.RandomizedTTestEstimator",
        "mean_test_benchmark",
    ),
    MethodRecord(
        "permutation_mean_test",
        MethodCategory.HYPOTHESIS_TEST,
        MethodStatus.IMPLEMENTED,
        "Monte Carlo randomization test for the difference in means.",
        "varwar.estimators.PermutationMeanTestEstimator",
        "mean_test_benchmark",
    ),
    MethodRecord(
        "anti_nuclear_ks_test",
        MethodCategory.HYPOTHESIS_TEST,
        MethodStatus.PLACEHOLDER_RND,
        "Tie-aware KS candidate with treatment-blind controlled de-tying and permutation-calibrated inference.",
        "varwar.estimators.PlaceholderEstimator",
        "distribution_test_backlog",
    ),
    MethodRecord(
        "cuped",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Configurable one-covariate CUPED; the primary local policy estimates theta on the current control.",
        "varwar.transforms.CupedTransformer",
    ),
    MethodRecord(
        "cuped_control",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "CUPED policy estimating theta from the current experiment control and applying it to both arms.",
        "varwar.transforms.CupedTransformer",
        "primary_pipeline_primitive",
    ),
    MethodRecord(
        "cupac_linear",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Temporal linear/ridge CUPAC trained on lag2 features to predict the lag1 pre-target within the available dataset.",
        "varwar.transforms.LinearCupacTransformer",
    ),
    MethodRecord(
        "linear_intime_adjustment",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Control-fitted linear second-stage target adjustment for pre-approved in-time covariates.",
        "varwar.transforms.LinearInTimeAdjustmentTransformer",
        "primary_pipeline_primitive",
    ),
    MethodRecord(
        "cupac_quadratic_A",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Temporal CUPAC over zone A with linear, squared, and pairwise interaction columns.",
        "varwar.transforms.QuadraticCupacTransformer",
        "exploratory_pipeline_primitive",
    ),
    MethodRecord(
        "nonlinear_cupac",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.PLACEHOLDER_RND,
        "General nonlinear temporal CUPAC beyond the implemented bounded quadratic model.",
        "varwar.placeholders.PlaceholderTargetTransformer",
    ),
    MethodRecord(
        "robust_cuped",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.PLACEHOLDER_RND,
        "CUPED-family adjustment with a robustly estimated coefficient for heavy-tailed experiment data.",
        "varwar.placeholders.PlaceholderTargetTransformer",
    ),
    MethodRecord(
        "winsorized",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Control-quantile clipping of the target; explicitly changes the target metric.",
        "varwar.transforms.WinsorizeTransformer",
    ),
    MethodRecord(
        "trimming",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.PLACEHOLDER_RND,
        "Control-defined tail removal awaiting a composable missing-row and inference contract.",
        "varwar.placeholders.PlaceholderTargetTransformer",
    ),
    MethodRecord(
        "bayesian_cupac",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "Bayesian prognostic adjustment awaiting a posterior target, model, and decision contract.",
        "varwar.placeholders.PlaceholderTargetTransformer",
    ),
    MethodRecord(
        "ratio_linearization",
        MethodCategory.METRIC_TRANSFORMER,
        MethodStatus.PLACEHOLDER_RND,
        "Composable linearization of a ratio metric before CUPED/CUPAC and a compatible base estimator.",
        "varwar.placeholders.PlaceholderMetricTransformer",
    ),
    MethodRecord(
        "numeric_feature_binner",
        MethodCategory.FEATURE_ADAPTER,
        MethodStatus.IMPLEMENTED,
        "Stable quantile, uniform, or custom categorical bins for numeric features.",
        "varwar.feature_adapters.NumericFeatureBinner",
    ),
    MethodRecord(
        "atomic_strata",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.IMPLEMENTED,
        "No-reduction baseline retaining every observed atomic category combination.",
        "varwar.stratification_benchmark._baseline_partitions",
        "stratification_preprocessor_race",
    ),
    MethodRecord(
        "random_reduction",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.IMPLEMENTED,
        "Balanced random grouping baseline for atomic strata.",
        "varwar.stratification_benchmark._baseline_partitions",
        "stratification_preprocessor_race",
    ),
    MethodRecord(
        "mean_sorted_reduction",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.IMPLEMENTED,
        "Deterministic grouping baseline ordered by historical atomic means.",
        "varwar.stratification_benchmark._baseline_partitions",
        "stratification_preprocessor_race",
    ),
    MethodRecord(
        "genetic_strata_reducer",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.IMPLEMENTED,
        "Genetic reduction of atomic strata into a fixed number of non-empty macrostrata.",
        "varwar.stratification.GeneticStrataReducer",
        "stratification_preprocessor_race",
    ),
    MethodRecord(
        "prognostic_score_block_randomization",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.PLACEHOLDER_RND,
        "Randomized assignment within blocks formed by an honest pre-treatment prognostic score.",
        "varwar.placeholders.PlaceholderDesignPreprocessor",
        "prospective_design_backlog",
    ),
    MethodRecord(
        "calibration_reweighting",
        MethodCategory.ESTIMATOR,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "Population calibration estimator requiring external population totals and a variance contract.",
        "varwar.estimators.PlaceholderEstimator",
        "population_generalization_backlog",
    ),
    MethodRecord(
        "optimal_transport_matching",
        MethodCategory.ESTIMATOR,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "Optimal-transport matching candidate for observational data; matching and inference contracts are not fixed.",
        "varwar.estimators.PlaceholderEstimator",
        "observational_backlog",
    ),
    MethodRecord(
        "optimal_transport_homogeneity_test",
        MethodCategory.HYPOTHESIS_TEST,
        MethodStatus.PLACEHOLDER_RND,
        "Optimal-transport two-sample homogeneity test; calibration and a scalable solver contract remain to be fixed.",
        "varwar.estimators.PlaceholderEstimator",
        "distribution_test_backlog",
    ),
    MethodRecord(
        "hypex_matcher",
        MethodCategory.CONFIGURATION_PROFILE,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "Possible HypEx matching integration profile; overlaps the local ATT matcher until parity rules are fixed.",
        "varwar.estimators.PlaceholderEstimator",
        "integration_backlog",
    ),
    MethodRecord(
        "optimal_transport_check",
        MethodCategory.HYPOTHESIS_TEST,
        MethodStatus.EXCLUDED,
        "Excluded orphan label: no unique statistic or decision rule distinguishes it from defined distribution tests.",
        "not_applicable",
        "excluded",
    ),
    MethodRecord(
        "ab_proportional",
        MethodCategory.OPTIMIZATION_OBJECTIVE,
        MethodStatus.IMPLEMENTED,
        "Internal GA objective for proportional A/B traffic; not an estimator or group-allocation method.",
        "varwar.stratification.ABProportionalVarianceObjective",
        "internal_stratification_objective",
    ),
    MethodRecord(
        "cupac_linear_A",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Temporal CUPAC baseline using only zone A pre-treatment lag1/lag2 features.",
        "varwar.transforms.ZonedCupacTransformer",
        "primary_pipeline_primitive",
    ),
    MethodRecord(
        "cupac_linear_A_plus_B",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Temporal CUPAC A followed by a control-fitted linear second stage over expert-safe zone B.",
        "varwar.transforms.ZonedCupacTransformer",
        "primary_pipeline_primitive",
    ),
    MethodRecord(
        "cupac_linear_A_plus_B_plus_C",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "CUPAC A, then linear B, then linear C admitted by exclusion, balance, missingness, and distribution gates.",
        "varwar.transforms.ZonedCupacTransformer",
        "primary_pipeline_primitive",
    ),
    MethodRecord(
        "bonferroni",
        MethodCategory.INFERENCE_CORRECTION,
        MethodStatus.IMPLEMENTED,
        "Single-step family-wise error rate correction.",
        "varwar.corrections.bonferroni",
    ),
    MethodRecord(
        "holm_bonferroni",
        MethodCategory.INFERENCE_CORRECTION,
        MethodStatus.IMPLEMENTED,
        "Step-down Bonferroni family-wise error rate correction.",
        "varwar.corrections.holm_bonferroni",
    ),
    MethodRecord(
        "romano_wolf_stepdown",
        MethodCategory.INFERENCE_CORRECTION,
        MethodStatus.IMPLEMENTED,
        "Step-down max-statistic correction over caller-supplied joint bootstrap statistics.",
        "varwar.corrections.romano_wolf_stepdown",
    ),
    MethodRecord(
        "welch",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED,
        "Unadjusted primary race pipeline containing the Welch difference-in-means estimator.",
        "varwar.pipelines.build_full_traffic_pipelines",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "hypex_cuped",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Native PyPI HypEx CUPED followed by its default Welch A/B inference.",
        "varwar.estimators.HypExCupedEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "cuped_control__hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Control-fitted local CUPED followed by the HypEx Welch A/B baseline.",
        "varwar.core.WrappedEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "hypex_cupac_temporal",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "HypEx temporal CUPAC followed by HypEx A/B; VarWar forces a linear CUPAC model.",
        "varwar.estimators.HypExCupacEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "cupac_linear_A__hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Temporal local CUPAC A followed by the HypEx Welch A/B baseline.",
        "varwar.core.WrappedEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "cupac_linear_A_plus_B__hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Temporal CUPAC A and linear B second-stage followed by HypEx Welch.",
        "varwar.core.WrappedEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "cupac_linear_A_plus_B_plus_C__hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Temporal CUPAC A, linear B, and gated linear C followed by HypEx Welch.",
        "varwar.core.WrappedEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "cupac_quadratic_A__hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Quadratic temporal CUPAC A followed by HypEx Welch.",
        "varwar.core.WrappedEstimator",
        "exploratory_pipeline_race",
    ),
    MethodRecord(
        "winsorized__hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Control-defined winsorization followed by HypEx Welch inference for the changed metric.",
        "varwar.core.WrappedEstimator",
        "robustness_pipeline_race",
    ),
    MethodRecord(
        "ga_strata__stratified",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED,
        "Genetic strata reduction followed by a weighted stratified ATE estimator.",
        "varwar.pipelines.build_stratified_pipelines",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "ga_strata__cuped_control__stratified",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED,
        "Genetic strata reduction, control-fitted CUPED, and a weighted stratified ATE estimator.",
        "varwar.pipelines.build_stratified_pipelines",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "ga_strata__cupac_linear_A__stratified",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED,
        "Genetic strata reduction, temporal CUPAC A, and a weighted stratified ATE estimator.",
        "varwar.pipelines.build_stratified_pipelines",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "ga_strata__cupac_linear_A_plus_B__stratified",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED,
        "Genetic strata reduction, CUPAC A and linear B, then a weighted stratified ATE estimator.",
        "varwar.pipelines.build_stratified_pipelines",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "ga_strata__cupac_linear_A_plus_B_plus_C__stratified",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED,
        "Genetic strata reduction, CUPAC A, linear B and gated C, then a weighted stratified ATE estimator.",
        "varwar.pipelines.build_stratified_pipelines",
        "primary_pipeline_race",
    ),
)

_INDEX = {record.method_id: record for record in _METHODS}
if len(_INDEX) != len(_METHODS):
    raise RuntimeError("Method catalog contains duplicate IDs.")


def method_catalog(
    category: MethodCategory | None = None,
    status: MethodStatus | None = None,
) -> tuple[MethodRecord, ...]:
    """Вернуть неизменяемые записи каталога с необязательной фильтрацией."""

    return tuple(
        record
        for record in _METHODS
        if (category is None or record.category == category)
        and (status is None or record.status == status)
    )


def get_method_record(method_id: str) -> MethodRecord:
    """Вернуть запись по ID или явно сообщить о неизвестном методе."""

    try:
        return _INDEX[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown VarWar method: {method_id}") from exc
