from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MethodCategory(str, Enum):
    """Role of a public R&D method in an experiment pipeline."""

    ESTIMATOR = "Estimator"
    TARGET_TRANSFORMER = "TargetTransformer"
    DESIGN_PREPROCESSOR = "DesignPreprocessor"
    FEATURE_ADAPTER = "FeatureAdapter"
    INFERENCE_CORRECTION = "InferenceCorrection"
    COMPOSITE_PIPELINE = "CompositePipeline"


class MethodStatus(str, Enum):
    """Implementation maturity recorded without implying production readiness."""

    IMPLEMENTED = "implemented"
    IMPLEMENTED_INTEGRATION = "implemented_integration"
    PLACEHOLDER_RND = "placeholder_rnd"
    PLACEHOLDER_NEEDS_SPEC = "placeholder_needs_spec"


@dataclass(frozen=True)
class MethodRecord:
    """Searchable metadata for one user-facing R&D method or race participant."""

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
        }


_METHODS = (
    MethodRecord(
        "welch_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference in means with Welch inference; canonical no-adjustment baseline.",
        "varwar.estimators.DifferenceInMeansEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "hypex_ab",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "PyPI HypEx A/B parity baseline with an explicit local fallback.",
        "varwar.estimators.HypExABTestEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "pooled_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference in means under the equal-variance assumption.",
        "varwar.estimators.DifferenceInMeansEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "ks_test",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Exact two-sample Kolmogorov-Smirnov distribution test.",
        "varwar.estimators.KSTestEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "approx_ks_fast",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Histogram approximation of the two-sample KS statistic.",
        "varwar.estimators.ApproxKSTestEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "energy_test",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Energy-distance two-sample test with bounded-memory permutations.",
        "varwar.estimators.EnergyDistanceTestEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "randomized_t_test",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Randomized hedgehog-weighted t statistic.",
        "varwar.estimators.RandomizedTTestEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "permutation_mean_test",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Monte Carlo randomization test for the mean difference.",
        "varwar.estimators.PermutationMeanTestEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "regression_adjustment",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "OLS treatment coefficient with HC1 robust inference.",
        "varwar.estimators.RegressionAdjustmentEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "ml_rate",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Cross-fitted ridge nuisance prediction with robust treatment adjustment.",
        "varwar.estimators.MLRateEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "nearest_neighbor_matching",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Nearest-neighbor covariate matching ATT baseline.",
        "varwar.estimators.NearestNeighborMatchingEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "ratio_delta",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference of ratio metrics with delta-method inference.",
        "varwar.estimators.RatioDeltaMethodEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "stratified_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Population-weighted mean of within-stratum effects.",
        "varwar.estimators.StratifiedDifferenceInMeansEstimator",
        "both_races",
    ),
    MethodRecord(
        "variance_weighted_stratified",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Inverse-variance combination of within-stratum effects.",
        "varwar.estimators.VarianceWeightedStratifiedEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "trimmed_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference of trimmed means for heavy-tailed metrics.",
        "varwar.estimators.TrimmedMeanEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "winsorized_mean_diff",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Difference of winsorized means for heavy-tailed metrics.",
        "varwar.estimators.WinsorizedMeanEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "wald_late",
        MethodCategory.ESTIMATOR,
        MethodStatus.IMPLEMENTED,
        "Wald ratio estimate of the local average treatment effect.",
        "varwar.estimators.WaldEstimator",
        "estimator_race",
    ),
    MethodRecord(
        "anti_nuclear_ks_test",
        MethodCategory.ESTIMATOR,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "Unidentified test name from the source worksheet.",
        "varwar.estimators.PlaceholderEstimator",
    ),
    MethodRecord(
        "hypex_matcher",
        MethodCategory.ESTIMATOR,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "HypEx matching candidate awaiting a fixed matching and inference policy.",
        "varwar.estimators.PlaceholderEstimator",
    ),
    MethodRecord(
        "calibration_reweighting",
        MethodCategory.ESTIMATOR,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "Calibration estimator awaiting target totals and variance policy.",
        "varwar.estimators.PlaceholderEstimator",
    ),
    MethodRecord(
        "optimal_transport_check",
        MethodCategory.ESTIMATOR,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "Distribution diagnostic awaiting a precise estimand and test contract.",
        "varwar.estimators.PlaceholderEstimator",
    ),
    MethodRecord(
        "optimal_transport_matching",
        MethodCategory.ESTIMATOR,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "Transport matching awaiting distance, regularization, and inference policy.",
        "varwar.estimators.PlaceholderEstimator",
    ),
    MethodRecord(
        "cuped",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Single-covariate pre-experiment target adjustment.",
        "varwar.transforms.CupedTransformer",
    ),
    MethodRecord(
        "cupac_linear",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Linear or ridge prognostic target adjustment.",
        "varwar.transforms.LinearCupacTransformer",
    ),
    MethodRecord(
        "sklearn_cupac_A",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "CUPAC configured with pre-treatment features only.",
        "varwar.feature_sets.default_cupac_feature_sets",
    ),
    MethodRecord(
        "sklearn_cupac_A_plus_B",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "CUPAC with pre-treatment and expert-safe in-time features.",
        "varwar.feature_sets.default_cupac_feature_sets",
    ),
    MethodRecord(
        "sklearn_cupac_A_plus_B_plus_C",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "CUPAC with pre-treatment, safe, and balance-gated feature groups.",
        "varwar.feature_sets.default_cupac_feature_sets",
    ),
    MethodRecord(
        "winsorized",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.IMPLEMENTED,
        "Reference-quantile clipping of the target.",
        "varwar.transforms.WinsorizeTransformer",
    ),
    MethodRecord(
        "bayesian_cupac",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.PLACEHOLDER_NEEDS_SPEC,
        "Bayesian prognostic adjustment awaiting a model and posterior contract.",
        "varwar.placeholders.PlaceholderTargetTransformer",
    ),
    MethodRecord(
        "nonlinear_cupac",
        MethodCategory.TARGET_TRANSFORMER,
        MethodStatus.PLACEHOLDER_RND,
        "Honest nonlinear prognostic adjustment trained on reference outcomes.",
        "varwar.placeholders.PlaceholderTargetTransformer",
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
        "No-reduction stratification baseline over all atomic combinations.",
        "varwar.stratification_benchmark._baseline_partitions",
        "stratification_race",
    ),
    MethodRecord(
        "random_reduction",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.IMPLEMENTED,
        "Balanced random grouping baseline for atomic strata.",
        "varwar.stratification_benchmark._baseline_partitions",
        "stratification_race",
    ),
    MethodRecord(
        "mean_sorted_reduction",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.IMPLEMENTED,
        "Deterministic grouping baseline ordered by historical atomic means.",
        "varwar.stratification_benchmark._baseline_partitions",
        "stratification_race",
    ),
    MethodRecord(
        "genetic_strata_reducer",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.IMPLEMENTED,
        "Genetic reduction of atomic strata into a fixed number of macrostrata.",
        "varwar.stratification.GeneticStrataReducer",
        "stratification_race",
    ),
    MethodRecord(
        "ab_proportional",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.IMPLEMENTED,
        "Variance objective for natural proportional A/B traffic.",
        "varwar.stratification.ABProportionalVarianceObjective",
        "stratification_race",
    ),
    MethodRecord(
        "genetic_strata_ab",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.IMPLEMENTED,
        "Genetic strata reducer configured with the proportional A/B objective.",
        "varwar.stratification_benchmark.run_stratification_race",
        "stratification_race",
    ),
    MethodRecord(
        "prognostic_score_block_randomization",
        MethodCategory.DESIGN_PREPROCESSOR,
        MethodStatus.PLACEHOLDER_RND,
        "Randomized treatment assignment within blocks formed by an honest prognostic score.",
        "varwar.placeholders.PlaceholderDesignPreprocessor",
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
        "Step-down max-statistic adjustment over caller-supplied bootstrap statistics.",
        "varwar.corrections.romano_wolf_stepdown",
    ),
    MethodRecord(
        "cuped+hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "CUPED target adjustment followed by the HypEx A/B baseline.",
        "varwar.core.WrappedEstimator",
        "both_races",
    ),
    MethodRecord(
        "hypex_cupac",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "PyPI HypEx CUPAC parity slot with an explicit local fallback.",
        "varwar.estimators.HypExCupacEstimator",
        "both_races",
    ),
    MethodRecord(
        "sklearn_cupac_A+hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Pre-treatment-only CUPAC followed by the HypEx A/B baseline.",
        "varwar.core.WrappedEstimator",
        "both_races",
    ),
    MethodRecord(
        "sklearn_cupac_A_plus_B+hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Safe in-time CUPAC variant followed by the HypEx A/B baseline.",
        "varwar.core.WrappedEstimator",
        "both_races",
    ),
    MethodRecord(
        "sklearn_cupac_A_plus_B_plus_C+hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Balance-gated CUPAC variant followed by the HypEx A/B baseline.",
        "varwar.core.WrappedEstimator",
        "both_races",
    ),
    MethodRecord(
        "winsorized+hypex_ab",
        MethodCategory.COMPOSITE_PIPELINE,
        MethodStatus.IMPLEMENTED_INTEGRATION,
        "Reference winsorization followed by the HypEx A/B baseline.",
        "varwar.core.WrappedEstimator",
        "estimator_race",
    ),
)

_INDEX = {record.method_id: record for record in _METHODS}
if len(_INDEX) != len(_METHODS):
    raise RuntimeError("Method catalog contains duplicate IDs.")


def method_catalog(
    category: MethodCategory | None = None,
    status: MethodStatus | None = None,
) -> tuple[MethodRecord, ...]:
    """Return immutable method metadata, optionally filtered by category/status."""

    return tuple(
        record
        for record in _METHODS
        if (category is None or record.category == category)
        and (status is None or record.status == status)
    )


def get_method_record(method_id: str) -> MethodRecord:
    """Return one catalog record and fail explicitly for an unknown method ID."""

    try:
        return _INDEX[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown VarWar method: {method_id}") from exc
