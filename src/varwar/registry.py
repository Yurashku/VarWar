from __future__ import annotations

from varwar.core import ExperimentSpec, WrappedEstimator
from varwar.estimators import (
    ApproxKSTestEstimator,
    DifferenceInMeansEstimator,
    EnergyDistanceTestEstimator,
    HypExABTestEstimator,
    HypExCupacEstimator,
    KSTestEstimator,
    MLRateEstimator,
    NearestNeighborMatchingEstimator,
    PlaceholderEstimator,
    PermutationMeanTestEstimator,
    RandomizedTTestEstimator,
    RatioDeltaMethodEstimator,
    RegressionAdjustmentEstimator,
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
from varwar.transforms import CupedTransformer, LinearCupacTransformer, WinsorizeTransformer


def placeholder_estimators() -> list[PlaceholderEstimator]:
    return [
        PlaceholderEstimator(
            "anti_nuclear_ks_test",
            "needs a treatment-blind de-tying rule, scale rule, and "
            "permutation-calibrated null distribution",
        ),
        PlaceholderEstimator(
            "hypex_matcher",
            "overlaps the local ATT matcher until estimand, distance, "
            "replacement, and parity rules are fixed",
        ),
        PlaceholderEstimator(
            "calibration_reweighting",
            "requires external population totals, calibration constraints, "
            "target population, and variance policy",
        ),
        PlaceholderEstimator(
            "optimal_transport_check",
            "excluded because no unique statistic or decision rule is defined",
        ),
        PlaceholderEstimator(
            "optimal_transport_matching",
            "needs matching distance, regularization, and inference policy",
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
            "nonlinear_cupac",
            "needs an honest reference-fit model contract, tuning policy, and "
            "resource-bounded model family",
        ),
        PlaceholderTargetTransformer(
            "robust_cuped",
            "needs a robust coefficient estimator, centering policy, "
            "honest fit contract, and valid inference tests",
        ),
        PlaceholderTargetTransformer(
            "trimming",
            "needs reference thresholds plus an explicit row-filtering and "
            "missing-row inference contract",
        ),
    ]


def placeholder_metric_transformers() -> list[PlaceholderMetricTransformer]:
    """Return non-executable transformations of source metric definitions."""

    return [
        PlaceholderMetricTransformer(
            "ratio_linearization",
            "needs a reference ratio, scale restoration rule, and a contract "
            "for composition with target transformers and base estimators",
        )
    ]


def placeholder_design_preprocessors() -> list[PlaceholderDesignPreprocessor]:
    """Return experiment-design candidates kept outside estimator races."""

    return [
        PlaceholderDesignPreprocessor(
            "prognostic_score_block_randomization",
            "needs a randomized block-construction and assignment contract, "
            "allocation ratios, and matching randomization-aware inference",
        )
    ]


def default_estimators(
    include_optional: bool = True,
    spec: ExperimentSpec | None = None,
    include_unsafe_demo: bool = False,
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
        WrappedEstimator(CupedTransformer(), HypExABTestEstimator()),
        HypExCupacEstimator(cupac_features=spec.pre_treatment_features()),
    ]
    for feature_set in default_cupac_feature_sets(
        spec,
        include_unsafe_demo=include_unsafe_demo,
    ):
        if feature_set.columns:
            estimators.append(
                WrappedEstimator(
                    LinearCupacTransformer(
                        feature_cols=feature_set.columns,
                        name=feature_set.name,
                    ),
                    HypExABTestEstimator(),
                    name=f"{feature_set.name}+hypex_ab",
                )
            )
    estimators.extend(
        [
            WrappedEstimator(WinsorizeTransformer(), HypExABTestEstimator()),
            DifferenceInMeansEstimator(name="pooled_mean_diff", variance="pooled"),
            KSTestEstimator(),
            ApproxKSTestEstimator(),
            EnergyDistanceTestEstimator(n_resamples=199),
            RandomizedTTestEstimator(),
            RegressionAdjustmentEstimator(),
            MLRateEstimator(),
            NearestNeighborMatchingEstimator(),
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
