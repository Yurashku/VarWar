from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MethodCategory(str, Enum):
    """Роль публичного метода в экспериментальном конвейере."""

    ESTIMATOR = "Estimator"
    POPULATION_GENERALIZATION_ESTIMATOR = "PopulationGeneralizationEstimator"
    HYPOTHESIS_TEST = "HypothesisTest"
    TARGET_TRANSFORMER = "TargetTransformer"
    METRIC_TRANSFORMER = "MetricTransformer"
    DESIGN_PREPROCESSOR = "DesignPreprocessor"
    RANDOMIZATION_STRUCTURE_BUILDER = "RandomizationStructureBuilder"
    ASSIGNMENT_MECHANISM = "AssignmentMechanism"
    ANALYSIS_METHOD = "AnalysisMethod"
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
class MethodCapability:
    """One scientifically valid way to use a method in a benchmark."""

    design_mode: str
    information_mode: str
    estimand_id: str
    temporal_mode: str = "pre_treatment_only"
    notes: str = ""


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

    @property
    def capabilities(self) -> tuple[MethodCapability, ...]:
        return _CAPABILITY_MAP.get(self.method_id, ())


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
        MethodStatus.IMPLEMENTED_LEGACY,
        "Unbiased hedgehog-weighted mean difference; retained as a normal-approximation research control, not an efficiency method.",
        "varwar.estimators.RandomizedWeightedMeanDiffEstimator",
        "research_only_not_primary_race",
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
        "Difference-in-means estimator with coherent within-arm studentized bootstrap-t inference.",
        "varwar.estimators.BootstrapMeanDiffEstimator",
        "primary_pipeline_race",
    ),
    MethodRecord(
        "regression_adjustment",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Lin-style centered regression adjustment with treatment interactions and HC2 heteroskedasticity-robust inference.",
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
        "population_weighted_stratified",
        MethodCategory.POPULATION_GENERALIZATION_ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Within-stratum A/B effects averaged with known target-population stratum shares.",
        "varwar.estimators.PopulationWeightedStratifiedEstimator",
        "population_information_race",
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
        "Nearest-neighbour matching estimator with an explicit symmetric ATE mode for randomized experiments and ATT compatibility outside the active race.",
        "varwar.estimators.NearestNeighborMatchingEstimator",
        "observational_specialized",
    ),
    MethodRecord(
        "hypex_matching",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "PyPI HypEx two-sided matching integration returning its bias-adjusted ATE.",
        "varwar.estimators.HypExMatchingEstimator",
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
        MethodStatus.IMPLEMENTED,
        "Treatment-blind deterministic de-tying used inside mixed-balance rerandomization diagnostics.",
        "varwar.assignment.RerandomizationMixedBalance",
        "assignment_diagnostic",
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
        MethodStatus.IMPLEMENTED,
        "Temporal CUPAC A selecting ridge, quadratic ridge, random forest, extra trees or histogram gradient boosting by pre-period CV.",
        "varwar.advanced_transforms.AutoMLTemporalCupacTransformer",
    ),
    MethodRecord(
        "robust_cuped",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Historical CUPED whose slope is estimated by Huber regression only from the lag2-to-lag1 transition.",
        "varwar.advanced_transforms.RobustHistoricalCupedTransformer",
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
        MethodStatus.EXCLUDED,
        "Old transformer label replaced by the Yuen trimmed-effect AnalysisMethod with its own estimand and inference.",
        "varwar.robust_estimators.YuenTrimmedMeanEstimator",
        "excluded_alias",
    ),
    MethodRecord(
        "isolation_forest_trimming",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Treatment-blind Isolation-Forest trimming with explicit contamination and changed-metric contract.",
        "varwar.transforms.IsolationForestTargetTransformer(method='trim')",
        "robustness_backlog",
    ),
    MethodRecord(
        "isolation_forest_winsorization",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Treatment-blind Isolation-Forest winsorization with explicit contamination and changed-metric contract.",
        "varwar.transforms.IsolationForestTargetTransformer(method='winsorize')",
        "robustness_backlog",
    ),
    MethodRecord(
        "bayesian_cupac",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.EXCLUDED,
        "Historical alias; use panel_prior_cupac for the implemented panel-summary prior contract.",
        "varwar.placeholders.PlaceholderTargetTransformer",
    ),
    MethodRecord(
        "panel_prior_cupac_A",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Temporal CUPAC A with a discounted prior learned from historical-panel sufficient statistics.",
        "varwar.transforms.PanelPriorCupacTransformer",
        "panel_prior_pipeline_primitive",
    ),
    MethodRecord(
        "panel_prior_cupac_A_plus_B",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Panel-prior CUPAC A followed by a separate control-fitted in-time B stage.",
        "varwar.transforms.PanelPriorZonedCupacTransformer",
        "panel_prior_pipeline_primitive",
    ),
    MethodRecord(
        "panel_prior_cupac_A_plus_B_plus_C",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Panel-prior CUPAC A, control-fitted B and balance-gated C stages.",
        "varwar.transforms.PanelPriorZonedCupacTransformer",
        "panel_prior_pipeline_primitive",
    ),
    MethodRecord(
        "ratio_linearization",
        MethodCategory.METRIC_TRANSFORMER,
        MethodStatus.EXCLUDED,
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
        MethodCategory.RANDOMIZATION_STRUCTURE_BUILDER,
        MethodStatus.IMPLEMENTED,
        "Frozen ridge prognostic score and quantile blocks fitted only on pre-treatment data.",
        "varwar.randomization_structure.PrognosticScoreBlockBuilder",
        "active_rct_design",
    ),
    MethodRecord(
        "calibration_reweighting",
        MethodCategory.POPULATION_GENERALIZATION_ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Entropy-calibrated A/B means reproducing known target-population covariate moments within each arm.",
        "varwar.estimators.CalibrationWeightedATEEstimator",
        "population_information_race",
    ),
    MethodRecord(
        "optimal_transport_matching",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Optimal-transport matching estimator with explicit symmetric ATE mode for randomized experiments; observational use remains specialized.",
        "varwar.estimators.OptimalTransportMatchingEstimator",
        "observational_specialized",
    ),
    MethodRecord(
        "optimal_transport_homogeneity_test",
        MethodCategory.HYPOTHESIS_TEST,
        MethodStatus.EXCLUDED,
        "Excluded: the active OT implementation is an ATE matching estimator, not a distribution test.",
        "not_applicable",
        "excluded",
    ),
    MethodRecord(
        "hypex_matcher",
        MethodCategory.CONFIGURATION_PROFILE,
        MethodStatus.EXCLUDED,
        "Historical alias replaced by the executable hypex_matching estimator.",
        "not_applicable",
        "excluded_alias",
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
        "panel_prior_cupac_A__hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Panel-prior CUPAC A followed by HypEx Welch A/B inference.",
        "varwar.pipelines.build_panel_prior_pipelines",
        "panel_prior_race",
    ),
    MethodRecord(
        "panel_prior_cupac_A_plus_B__hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Panel-prior CUPAC A, separate B stage and HypEx Welch A/B inference.",
        "varwar.pipelines.build_panel_prior_pipelines",
        "panel_prior_race",
    ),
    MethodRecord(
        "panel_prior_cupac_A_plus_B_plus_C__hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Panel-prior CUPAC A, B, gated C and HypEx Welch A/B inference.",
        "varwar.pipelines.build_panel_prior_pipelines",
        "panel_prior_race",
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
    MethodRecord("complete_randomization", MethodCategory.ASSIGNMENT_MECHANISM, MethodStatus.IMPLEMENTED, "Полная рандомизация с точными размерами групп.", "varwar.assignment.CompleteRandomization", "active_rct_design"),
    MethodRecord("stratified_randomization", MethodCategory.ASSIGNMENT_MECHANISM, MethodStatus.IMPLEMENTED, "Назначение внутри заранее построенных страт с точным общим размером treatment и рандомизированным округлением квот.", "varwar.assignment.StratifiedRandomization", "active_rct_design"),
    MethodRecord("rerandomization_mahalanobis", MethodCategory.ASSIGNMENT_MECHANISM, MethodStatus.IMPLEMENTED, "Рерандомизация по заранее откалиброванному расстоянию Махаланобиса.", "varwar.assignment.RerandomizationMahalanobis", "active_rct_design"),
    MethodRecord("rerandomization_max_smd", MethodCategory.ASSIGNMENT_MECHANISM, MethodStatus.IMPLEMENTED, "Рерандомизация по максимальному абсолютному SMD.", "varwar.assignment.RerandomizationMaxSMD", "active_rct_design"),
    MethodRecord("rerandomization_mixed_balance", MethodCategory.ASSIGNMENT_MECHANISM, MethodStatus.IMPLEMENTED, "Рерандомизация по числовому, категориальному и tie-aware распределительному балансу.", "varwar.assignment.RerandomizationMixedBalance", "active_rct_design"),
    MethodRecord("hypex_aa_best_of_k", MethodCategory.ASSIGNMENT_MECHANISM, MethodStatus.IMPLEMENTED_INTEGRATION, "Политика выбора сплита через официальный PyPI HypEx AATest.", "varwar.assignment.HypExAABestOfK", "integration_parity"),
    MethodRecord("prognostic_blocks", MethodCategory.RANDOMIZATION_STRUCTURE_BUILDER, MethodStatus.IMPLEMENTED, "Блоки по честному pre-treatment prognostic score.", "varwar.randomization_structure.PrognosticScoreBlockBuilder", "active_rct_design"),
    MethodRecord("cuped_historical", MethodCategory.TARGET_TRANSFORMER, MethodStatus.IMPLEMENTED, "CUPED с коэффициентом, зафиксированным по историческому переходу lag2 к lag1.", "varwar.target_registry._HistoricalCupedTransformer", "active_rct_analysis"),
    MethodRecord("cuped_pooled_theta", MethodCategory.TARGET_TRANSFORMER, MethodStatus.IMPLEMENTED, "CUPED с theta = Cov(Y, X)/Var(X) по текущему эксперименту (обе группы вместе); асимптотически не увеличивает дисперсию.", "varwar.transforms.CupedTransformer(fit_data='experiment_all')", "active_rct_analysis"),
    MethodRecord("cupac_linear_pooled_theta_A", MethodCategory.TARGET_TRANSFORMER, MethodStatus.IMPLEMENTED, "CUPAC A: ridge-модель lag2 -> lag1, затем theta = Cov(Y, g)/Var(g) по текущему эксперименту вместо фиксированного коэффициента 1.", "varwar.transforms.LinearCupacTransformer(theta_policy='pooled')", "active_rct_analysis"),
    MethodRecord("ml_rate__crossfit_gbm_hc2", MethodCategory.ANALYSIS_METHOD, MethodStatus.IMPLEMENTED, "ML-RATE: cross-fitted HistGradientBoosting-прогноз без treatment, затем Lin-регрессия на центрированный прогноз с взаимодействием и HC2, df = n - 4.", "varwar.estimators.MLRateEstimator(learner='hist_gbm')", "sample_ate"),
    MethodRecord("lin_regression__hc2_pate", MethodCategory.ANALYSIS_METHOD, MethodStatus.IMPLEMENTED, "Lin HC2 для population ATE при iid-выборке из целевой популяции: Var = HC2 + gamma' Sigma_X gamma / n.", "varwar.estimators.RegressionAdjustmentEstimator(pate_correction=True)", "population_ate"),
    MethodRecord("cuped_huber_historical", MethodCategory.TARGET_TRANSFORMER, MethodStatus.IMPLEMENTED, "Исторический CUPED с Huber-оценкой коэффициента.", "varwar.advanced_transforms.RobustHistoricalCupedTransformer", "active_rct_analysis"),
    MethodRecord("cupac_automl_A", MethodCategory.TARGET_TRANSFORMER, MethodStatus.IMPLEMENTED, "CUPAC A с автоматическим выбором pre-period ML-модели.", "varwar.advanced_transforms.AutoMLTemporalCupacTransformer", "active_rct_analysis"),
    MethodRecord("yuen_trimmed_effect", MethodCategory.ANALYSIS_METHOD, MethodStatus.IMPLEMENTED, "Разность усечённых средних с выводом Юэна; estimand отличается от ATE.", "varwar.robust_estimators.YuenTrimmedMeanEstimator", "robust_native"),
    MethodRecord("winsorized_effect_bootstrap", MethodCategory.ANALYSIS_METHOD, MethodStatus.IMPLEMENTED, "Разность винзоризированных средних с полным переобучением границ в bootstrap.", "varwar.robust_estimators.FullRefitWinsorizedMeanEstimator", "robust_native"),
    MethodRecord("median_of_means_ate", MethodCategory.ANALYSIS_METHOD, MethodStatus.IMPLEMENTED, "Исследовательская устойчивая оценка исходного среднего через median-of-means.", "varwar.robust_estimators.MedianOfMeansEstimator", "robust_mean_ate"),
    MethodRecord("catoni_mean_ate", MethodCategory.ANALYSIS_METHOD, MethodStatus.IMPLEMENTED, "Исследовательская Catoni-оценка исходного среднего.", "varwar.robust_estimators.CatoniMeanEstimator", "robust_mean_ate"),
    MethodRecord(
        "population_weighted_stratified_ate__native",
        MethodCategory.ANALYSIS_METHOD,
        MethodStatus.IMPLEMENTED,
        "Стратифицированная оценка population ATE с долями страт из полной исторической панели.",
        "varwar.analysis_registry.build_analysis_method_registry",
        "population_ate",
    ),
    MethodRecord(
        "calibration_weighted_population_ate__native",
        MethodCategory.ANALYSIS_METHOD,
        MethodStatus.IMPLEMENTED,
        "Оценка population ATE с калибровкой весов по известным моментам целевой популяции.",
        "varwar.analysis_registry.build_analysis_method_registry",
        "population_ate",
    ),
)

# The executable v3 registries are the source of truth for newly composed
# TargetTransformer and AnalysisMethod IDs.  Historical records above remain
# searchable, while missing current IDs are added without duplicating aliases.
from varwar.analysis_registry import build_analysis_method_registry as _build_analysis_registry
from varwar.target_registry import TARGET_REGISTRY as _TARGET_REGISTRY

_known_method_ids = {record.method_id for record in _METHODS}
_METHODS += tuple(
    MethodRecord(
        method_id,
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED if definition.available else MethodStatus.EXCLUDED,
        definition.title_ru,
        "varwar.target_registry.TARGET_REGISTRY",
        "active_rct_analysis" if definition.available else "unavailable_integration_alias",
    )
    for method_id, definition in _TARGET_REGISTRY.items()
    if method_id not in _known_method_ids
)
_known_method_ids = {record.method_id for record in _METHODS}
_METHODS += tuple(
    MethodRecord(
        method_id,
        MethodCategory.ANALYSIS_METHOD,
        MethodStatus.IMPLEMENTED_INTEGRATION if "hypex" in method_id else MethodStatus.IMPLEMENTED,
        "Зарегистрированная неделимая пара точечной оценки и допустимого статистического вывода.",
        "varwar.analysis_registry.build_analysis_method_registry",
        method.contract.ranking_track,
    )
    for method_id, method in _build_analysis_registry(
        resamples=99,
        prefer_pypi_hypex=False,
    ).items()
    if method_id not in _known_method_ids
)

_RCT_PRE_ESTIMATORS = {
    "welch_mean_diff",
    "hypex_ab",
    "permutation_mean_diff",
    "bootstrap_mean_diff",
    "regression_adjustment",
    "ml_rate",
    "nearest_neighbor_matching",
    "hypex_matching",
    "optimal_transport_matching",
    "stratified_mean_diff",
    "cuped_control",
    "cupac_linear_A",
}
_CAPABILITY_MAP: dict[str, tuple[MethodCapability, ...]] = {
    method_id: (
        MethodCapability("randomized", "experiment_only", "sample_ate", "pre_treatment_only"),
        MethodCapability("randomized", "panel_prior", "sample_ate", "pre_treatment_only"),
    )
    for method_id in _RCT_PRE_ESTIMATORS
}
_CAPABILITY_MAP.update({
    "cupac_linear_A_plus_B": (
        MethodCapability("randomized", "experiment_only", "sample_ate", "post_treatment_allowed"),
        MethodCapability("randomized", "panel_prior", "sample_ate", "post_treatment_allowed"),
    ),
    "cupac_linear_A_plus_B_plus_C": (
        MethodCapability("randomized", "experiment_only", "sample_ate", "post_treatment_allowed"),
        MethodCapability("randomized", "panel_prior", "sample_ate", "post_treatment_allowed"),
    ),
    "panel_prior_cupac_A": (
        MethodCapability("randomized", "panel_prior", "sample_ate", "pre_treatment_only"),
    ),
    "panel_prior_cupac_A_plus_B": (
        MethodCapability("randomized", "panel_prior", "sample_ate", "post_treatment_allowed"),
    ),
    "panel_prior_cupac_A_plus_B_plus_C": (
        MethodCapability("randomized", "panel_prior", "sample_ate", "post_treatment_allowed"),
    ),
    "panel_prior_cupac_A__hypex_ab": (
        MethodCapability("randomized", "panel_prior", "sample_ate", "pre_treatment_only"),
    ),
    "panel_prior_cupac_A_plus_B__hypex_ab": (
        MethodCapability("randomized", "panel_prior", "sample_ate", "post_treatment_allowed"),
    ),
    "panel_prior_cupac_A_plus_B_plus_C__hypex_ab": (
        MethodCapability("randomized", "panel_prior", "sample_ate", "post_treatment_allowed"),
    ),
})

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
