from __future__ import annotations

from varwar.core import ExperimentSpec, WrappedEstimator
from varwar.estimators import (
    ApproxKSTestEstimator,
    BootstrapMeanDiffEstimator,
    CalibrationWeightedATEEstimator,
    DifferenceInMeansEstimator,
    EnergyDistanceTestEstimator,
    HypExABTestEstimator,
    HypExCupacEstimator,
    HypExCupedEstimator,
    HypExMatchingEstimator,
    KSTestEstimator,
    MLRateEstimator,
    NearestNeighborMatchingEstimator,
    OptimalTransportMatchingEstimator,
    PlaceholderEstimator,
    PermutationMeanTestEstimator,
    PermutationMeanDiffEstimator,
    RandomizedTTestEstimator,
    RatioDeltaMethodEstimator,
    RegressionAdjustmentEstimator,
    RandomizedWeightedMeanDiffEstimator,
    PopulationWeightedStratifiedEstimator,
    StratifiedDifferenceInMeansEstimator,
    TrimmedMeanEstimator,
    VarianceWeightedStratifiedEstimator,
    WaldEstimator,
    WinsorizedMeanEstimator,
)
from varwar.feature_sets import default_cupac_feature_sets
from varwar.placeholders import (
    PlaceholderDesignPreprocessor,
    PlaceholderMetricTransformer,
    PlaceholderTargetTransformer,
)
from varwar.transforms import (
    CupedTransformer,
    QuadraticCupacTransformer,
    WinsorizeTransformer,
    ZonedCupacTransformer,
)


def placeholder_estimators() -> list[PlaceholderEstimator]:
    return [
        PlaceholderEstimator(
            "optimal_transport_check",
            "excluded because no unique statistic or decision rule is defined",
        ),
        PlaceholderEstimator(
            "optimal_transport_homogeneity_test",
            "needs a scalable OT solver, a fixed statistic, and a "
            "permutation-calibration contract",
        ),
    ]


def placeholder_target_transformers() -> list[PlaceholderTargetTransformer]:
    """Return target-adjustment candidates that are intentionally non-executable."""

    return [
        PlaceholderTargetTransformer(
            "bayesian_cupac",
            "needs a Bayesian model and posterior decision rule",
        ),
        PlaceholderTargetTransformer(
            "trimming",
            "needs control-defined thresholds plus an explicit row-filtering "
            "and missing-row inference contract",
        ),
    ]


def placeholder_metric_transformers() -> list[PlaceholderMetricTransformer]:
    """Return non-executable transformations of source metric definitions."""

    return [
        PlaceholderMetricTransformer(
            "ratio_linearization",
            "needs a fixed baseline ratio, a scale restoration rule, and a "
            "contract for composition with target transformers and estimators",
        )
    ]


def placeholder_design_preprocessors() -> list[PlaceholderDesignPreprocessor]:
    """Return experiment-design candidates kept outside estimator races."""

    return []


def default_estimators(
    include_optional: bool = True,
    spec: ExperimentSpec | None = None,
) -> list:
    spec = spec or ExperimentSpec(
        covariate_cols=("y_pre", "x1", "x2"),
        pre_treatment_cols=("y_pre", "x1", "x2"),
        expert_safe_in_time_cols=("calendar_index", "market_index"),
        gated_in_time_cols=("operational_load",),
        unsafe_demo_cols=("post_outcome_proxy",),
    )
    estimators = [
        DifferenceInMeansEstimator(),
        HypExABTestEstimator(),
        HypExCupedEstimator(),
        WrappedEstimator(
            CupedTransformer(fit_data="experiment_control"),
            HypExABTestEstimator(),
            name="cuped_control__hypex_ab",
        ),
        HypExCupacEstimator(cupac_features=spec.pre_treatment_features()),
    ]
    for feature_set in default_cupac_feature_sets(spec):
        if spec.pre_treatment_features():
            estimators.append(
                WrappedEstimator(
                    ZonedCupacTransformer(policy=feature_set),
                    HypExABTestEstimator(),
                    name=f"{feature_set.method_id}__hypex_ab",
                )
            )
    estimators.append(
        WrappedEstimator(
            QuadraticCupacTransformer(
                feature_cols=spec.pre_treatment_features(),
            ),
            HypExABTestEstimator(),
            name="cupac_quadratic_A__hypex_ab",
        )
    )
    estimators.extend(
        [
            WrappedEstimator(
                WinsorizeTransformer(),
                HypExABTestEstimator(),
                name="winsorized__hypex_ab",
            ),
            DifferenceInMeansEstimator(name="pooled_mean_diff", variance="pooled"),
            KSTestEstimator(),
            ApproxKSTestEstimator(),
            EnergyDistanceTestEstimator(n_resamples=199),
            RandomizedTTestEstimator(),
            RandomizedWeightedMeanDiffEstimator(),
            PermutationMeanDiffEstimator(n_resamples=499),
            BootstrapMeanDiffEstimator(n_resamples=499),
            RegressionAdjustmentEstimator(),
            MLRateEstimator(),
            NearestNeighborMatchingEstimator(),
            HypExMatchingEstimator(),
            OptimalTransportMatchingEstimator(),
            TrimmedMeanEstimator(),
            WinsorizedMeanEstimator(),
        ]
    )
    if include_optional:
        estimators.extend(
            [
                RatioDeltaMethodEstimator(),
                StratifiedDifferenceInMeansEstimator(),
                VarianceWeightedStratifiedEstimator(),
                WaldEstimator(),
                PermutationMeanTestEstimator(n_resamples=499),
            ]
        )
    return estimators
