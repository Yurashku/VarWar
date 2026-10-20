from __future__ import annotations

from typing import Any

from varwar.analysis import (
    AnalysisMethod,
    CausalSafety,
    CompatibilityContract,
    InformationMode,
    LegacyEstimatorPointAdapter,
    NativeEstimatorInference,
    TemporalAccess,
    TheoryGrade,
)
from varwar.estimators import (
    CalibrationWeightedATEEstimator,
    HypExABTestEstimator,
    HypExCupacEstimator,
    HypExCupedEstimator,
    HypExMatchingEstimator,
    MLRateEstimator,
    NearestNeighborMatchingEstimator,
    OptimalTransportMatchingEstimator,
    PopulationWeightedStratifiedEstimator,
    RandomizedWeightedMeanDiffEstimator,
    RegressionAdjustmentEstimator,
    StratifiedDifferenceInMeansEstimator,
    VarianceWeightedStratifiedEstimator,
)
from varwar.inference import (
    BootstrapTInference,
    DesignBasedNormalInference,
    DifferenceMeansPointEstimator,
    PooledStudentInference,
    SharpNullRandomizationInference,
    StudentizedRandomizationInference,
    WelchInference,
)
from varwar.robust_estimators import (
    CatoniMeanEstimator,
    FullRefitWinsorizedMeanEstimator,
    IsolationForestSensitivityEstimator,
    MedianOfMeansEstimator,
    YuenTrimmedMeanEstimator,
)


COMPLETE_ASSIGNMENTS = frozenset({"complete_randomization"})
RANDOMIZED_ASSIGNMENTS = frozenset(
    {
        "complete_randomization",
        "stratified_randomization",
        "rerandomization_mahalanobis",
        "rerandomization_max_smd",
        "rerandomization_mixed_balance",
        "hypex_aa_best_of_k",
    }
)
STRATIFIED_ASSIGNMENTS = frozenset(
    RANDOMIZED_ASSIGNMENTS
)


def _native_method(
    method_id: str,
    estimator: Any,
    *,
    estimand_id: str,
    track: str,
    grade: TheoryGrade,
    safety: CausalSafety,
    assignment_ids: frozenset[str] = RANDOMIZED_ASSIGNMENTS,
    information_modes: frozenset[InformationMode] = frozenset(InformationMode),
    temporal: TemporalAccess = TemporalAccess.PRE_TREATMENT_ONLY,
    primary: bool = False,
    inference_available: bool = True,
) -> AnalysisMethod:
    return AnalysisMethod(
        method_id=method_id,
        point_estimator=LegacyEstimatorPointAdapter(
            estimator,
            estimand_id=estimand_id,
            method_id=f"{method_id}_point",
        ),
        inference=NativeEstimatorInference(
            estimator,
            inference_id=f"{method_id}_native",
        ),
        contract=CompatibilityContract(
            estimand_id=estimand_id,
            ranking_track=track,
            theory_grade=grade,
            causal_safety=safety,
            information_modes=information_modes,
            minimum_temporal_access=temporal,
            assignment_ids=assignment_ids,
            inference_requirements=frozenset({"native"}),
            inference_available=inference_available,
            primary_ranking_eligible=primary,
        ),
    )


def build_analysis_method_registry(
    *,
    resamples: int = 999,
    random_state: int = 42,
    prefer_pypi_hypex: bool = True,
    population_weights: dict[str, float] | None = None,
    population_moments: dict[str, float] | None = None,
) -> dict[str, AnalysisMethod]:
    """Единый исполняемый реестр анализа без свободных inference-пар."""

    point = DifferenceMeansPointEstimator()
    methods = {
        "diff_means__welch": AnalysisMethod(
            "diff_means__welch",
            point,
            WelchInference(),
            CompatibilityContract(
                "sample_ate",
                "sample_ate",
                TheoryGrade.A,
                CausalSafety.SAFE,
                assignment_ids=RANDOMIZED_ASSIGNMENTS,
                inference_requirements=frozenset({"unequal_variance"}),
                primary_ranking_eligible=True,
            ),
        ),
        "diff_means__student_pooled": AnalysisMethod(
            "diff_means__student_pooled",
            point,
            PooledStudentInference(),
            CompatibilityContract(
                "sample_ate",
                "diagnostic_only",
                TheoryGrade.B,
                CausalSafety.SAFE,
                assignment_ids=RANDOMIZED_ASSIGNMENTS,
                inference_requirements=frozenset({"equal_variance"}),
            ),
        ),
        "diff_means__design_based_normal": AnalysisMethod(
            "diff_means__design_based_normal",
            point,
            DesignBasedNormalInference(),
            CompatibilityContract(
                "sample_ate",
                "sample_ate",
                TheoryGrade.A,
                CausalSafety.SAFE,
                assignment_ids=RANDOMIZED_ASSIGNMENTS,
                inference_requirements=frozenset({"design_based"}),
                primary_ranking_eligible=True,
            ),
        ),
        "diff_means__welch_studentized_randomization": AnalysisMethod(
            "diff_means__welch_studentized_randomization",
            point,
            StudentizedRandomizationInference(resamples, random_state),
            CompatibilityContract(
                "sample_ate",
                "sample_ate",
                TheoryGrade.A,
                CausalSafety.SAFE,
                assignment_ids=COMPLETE_ASSIGNMENTS,
                inference_requirements=frozenset({"design_aware", "full_refit"}),
                primary_ranking_eligible=True,
            ),
        ),
        "diff_means__bootstrap_t": AnalysisMethod(
            "diff_means__bootstrap_t",
            point,
            BootstrapTInference(resamples, random_state),
            CompatibilityContract(
                "sample_ate",
                "sample_ate",
                TheoryGrade.B,
                CausalSafety.SAFE,
                assignment_ids=COMPLETE_ASSIGNMENTS,
                inference_requirements=frozenset({"full_refit", "superpopulation"}),
            ),
        ),
        "diff_means__sharp_null_randomization": AnalysisMethod(
            "diff_means__sharp_null_randomization",
            point,
            SharpNullRandomizationInference(resamples, random_state),
            CompatibilityContract(
                "sample_ate",
                "diagnostic_only",
                TheoryGrade.B,
                CausalSafety.SAFE,
                assignment_ids=COMPLETE_ASSIGNMENTS,
                inference_requirements=frozenset({"sharp_null", "full_refit"}),
            ),
        ),
    }
    native = (
        _native_method(
            "lin_regression__hc2",
            RegressionAdjustmentEstimator(covariance_type="HC2"),
            estimand_id="sample_ate",
            track="sample_ate",
            grade=TheoryGrade.A,
            safety=CausalSafety.SAFE,
            primary=True,
        ),
        _native_method(
            "lin_regression__hc3",
            RegressionAdjustmentEstimator(covariance_type="HC3"),
            estimand_id="sample_ate",
            track="sample_ate",
            grade=TheoryGrade.A,
            safety=CausalSafety.SAFE,
            primary=True,
        ),
        _native_method(
            "ml_rate__crossfit_hc2",
            MLRateEstimator(),
            estimand_id="sample_ate",
            track="sample_ate",
            grade=TheoryGrade.B,
            safety=CausalSafety.SAFE,
            primary=True,
        ),
        _native_method(
            "ml_rate__crossfit_gbm_hc2",
            MLRateEstimator(
                name="ml_rate_gbm",
                learner="hist_gbm",
                seed=random_state,
            ),
            estimand_id="sample_ate",
            track="sample_ate",
            grade=TheoryGrade.B,
            safety=CausalSafety.SAFE,
            primary=True,
        ),
        _native_method(
            "lin_regression__hc2_pate",
            RegressionAdjustmentEstimator(
                name="regression_adjustment_pate",
                covariance_type="HC2",
                pate_correction=True,
            ),
            estimand_id="population_ate",
            track="population_ate",
            grade=TheoryGrade.B,
            safety=CausalSafety.CONDITIONAL,
            primary=True,
        ),
        _native_method(
            "hypex_ab__integration",
            HypExABTestEstimator(
                prefer_pypi=prefer_pypi_hypex,
                require_pypi=prefer_pypi_hypex,
            ),
            estimand_id="sample_ate",
            track="integration_parity",
            grade=TheoryGrade.B,
            safety=CausalSafety.SAFE,
        ),
        _native_method(
            "hypex_cuped__integration",
            HypExCupedEstimator(
                prefer_pypi=prefer_pypi_hypex,
                require_pypi=prefer_pypi_hypex,
            ),
            estimand_id="sample_ate",
            track="integration_parity",
            grade=TheoryGrade.C,
            safety=CausalSafety.CONDITIONAL,
        ),
        _native_method(
            "hypex_cupac_A__integration",
            HypExCupacEstimator(
                prefer_pypi=prefer_pypi_hypex,
                require_pypi=prefer_pypi_hypex,
            ),
            estimand_id="sample_ate",
            track="integration_parity",
            grade=TheoryGrade.C,
            safety=CausalSafety.CONDITIONAL,
        ),
        _native_method(
            "stratified_ate__native",
            StratifiedDifferenceInMeansEstimator(),
            estimand_id="sample_ate",
            track="sample_ate",
            grade=TheoryGrade.A,
            safety=CausalSafety.SAFE,
            assignment_ids=STRATIFIED_ASSIGNMENTS,
            primary=True,
        ),
        _native_method(
            "nearest_neighbor_matching_ate__reuse_aware",
            NearestNeighborMatchingEstimator(estimand="ate"),
            estimand_id="sample_ate",
            track="sample_ate",
            grade=TheoryGrade.B,
            safety=CausalSafety.CONDITIONAL,
        ),
        _native_method(
            "optimal_transport_matching_ate__point",
            OptimalTransportMatchingEstimator(estimand="ate"),
            estimand_id="sample_ate",
            track="diagnostic_only",
            grade=TheoryGrade.D,
            safety=CausalSafety.CONDITIONAL,
            inference_available=False,
        ),
        _native_method(
            "hypex_matching_ate__integration",
            HypExMatchingEstimator(),
            estimand_id="sample_ate",
            track="integration_parity",
            grade=TheoryGrade.C,
            safety=CausalSafety.CONDITIONAL,
        ),
        _native_method(
            "hedgehog_weighted_ate__native",
            RandomizedWeightedMeanDiffEstimator(),
            estimand_id="sample_ate",
            track="diagnostic_only",
            grade=TheoryGrade.C,
            safety=CausalSafety.SAFE,
        ),
        _native_method(
            "common_effect_inverse_variance_stratified__native",
            VarianceWeightedStratifiedEstimator(),
            estimand_id="common_effect",
            track="robust_native",
            grade=TheoryGrade.C,
            safety=CausalSafety.CONDITIONAL,
            assignment_ids=STRATIFIED_ASSIGNMENTS,
        ),
        _native_method(
            "trimmed_mean__yuen",
            YuenTrimmedMeanEstimator(),
            estimand_id="trimmed_mean_effect",
            track="robust_native",
            grade=TheoryGrade.A,
            safety=CausalSafety.SAFE,
        ),
        _native_method(
            "winsorized_mean__full_refit_bootstrap",
            FullRefitWinsorizedMeanEstimator(
                n_resamples=resamples,
                random_state=random_state,
            ),
            estimand_id="winsorized_mean_effect",
            track="robust_native",
            grade=TheoryGrade.B,
            safety=CausalSafety.SAFE,
        ),
        _native_method(
            "iforest_trimmed_mean__full_refit",
            IsolationForestSensitivityEstimator(
                method="trim",
                n_resamples=resamples,
                random_state=random_state,
            ),
            estimand_id="isolation_forest_trim_effect",
            track="diagnostic_only",
            grade=TheoryGrade.D,
            safety=CausalSafety.NOT_CAUSAL,
        ),
        _native_method(
            "iforest_winsorized_mean__full_refit",
            IsolationForestSensitivityEstimator(
                method="winsorize",
                n_resamples=resamples,
                random_state=random_state,
            ),
            estimand_id="isolation_forest_winsorize_effect",
            track="diagnostic_only",
            grade=TheoryGrade.D,
            safety=CausalSafety.NOT_CAUSAL,
        ),
        _native_method(
            "median_of_means_ate__bootstrap",
            MedianOfMeansEstimator(n_resamples=resamples, random_state=random_state),
            estimand_id="sample_ate",
            track="robust_mean_ate",
            grade=TheoryGrade.C,
            safety=CausalSafety.SAFE,
        ),
        _native_method(
            "catoni_mean_ate__bootstrap",
            CatoniMeanEstimator(n_resamples=resamples, random_state=random_state),
            estimand_id="sample_ate",
            track="robust_mean_ate",
            grade=TheoryGrade.C,
            safety=CausalSafety.SAFE,
        ),
    )
    methods.update({method.method_id: method for method in native})
    ot_point = LegacyEstimatorPointAdapter(
        OptimalTransportMatchingEstimator(estimand="ate"),
        estimand_id="sample_ate",
        method_id="optimal_transport_matching_ate_point",
    )
    methods["optimal_transport_matching_ate__sharp_null_ri"] = AnalysisMethod(
        method_id="optimal_transport_matching_ate__sharp_null_ri",
        point_estimator=ot_point,
        inference=SharpNullRandomizationInference(resamples, random_state),
        contract=CompatibilityContract(
            estimand_id="sample_ate",
            ranking_track="diagnostic_only",
            theory_grade=TheoryGrade.C,
            causal_safety=CausalSafety.CONDITIONAL,
            point_assignment_ids=RANDOMIZED_ASSIGNMENTS,
            assignment_ids=COMPLETE_ASSIGNMENTS,
            inference_requirements=frozenset({"sharp_null", "full_refit"}),
            primary_ranking_eligible=False,
        ),
    )
    if population_weights:
        method = _native_method(
            "population_weighted_stratified_ate__native",
            PopulationWeightedStratifiedEstimator(population_weights),
            estimand_id="population_ate",
            track="population_ate",
            grade=TheoryGrade.A,
            safety=CausalSafety.CONDITIONAL,
            assignment_ids=STRATIFIED_ASSIGNMENTS,
            information_modes=frozenset({InformationMode.PANEL_PRIOR}),
            primary=True,
        )
        methods[method.method_id] = method
    if population_moments:
        method = _native_method(
            "calibration_weighted_population_ate__native",
            CalibrationWeightedATEEstimator(population_moments),
            estimand_id="population_ate",
            track="population_ate",
            grade=TheoryGrade.C,
            safety=CausalSafety.CONDITIONAL,
            information_modes=frozenset({InformationMode.PANEL_PRIOR}),
        )
        methods[method.method_id] = method
    return methods
