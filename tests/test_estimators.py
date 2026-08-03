from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from varwar.benchmark import build_default_race, run_estimator_race
from varwar.corrections import bonferroni, holm_bonferroni, romano_wolf_stepdown
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
    KSTestEstimator,
    MLRateEstimator,
    NearestNeighborMatchingEstimator,
    OptimalTransportMatchingEstimator,
    PlaceholderEstimator,
    PermutationMeanDiffEstimator,
    RandomizedTTestEstimator,
    RandomizedWeightedMeanDiffEstimator,
    RatioDeltaMethodEstimator,
    PopulationWeightedStratifiedEstimator,
    StratifiedDifferenceInMeansEstimator,
    TrimmedMeanEstimator,
)
from varwar.synthetic import SyntheticConfig, generate_historical_controls, generate_synthetic_ab
from varwar.feature_sets import default_cupac_feature_sets
from varwar.transforms import (
    CupedTransformer,
    IsolationForestTargetTransformer,
    LinearCupacTransformer,
    ZonedCupacTransformer,
)


class EstimatorTests(unittest.TestCase):
    def test_difference_in_means_recovers_known_effect(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 1, 1, 1],
                "y": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
            }
        )
        result = DifferenceInMeansEstimator().estimate(data, ExperimentSpec())
        self.assertAlmostEqual(result.effect, 2.0)
        self.assertEqual(result.n_control, 3)
        self.assertEqual(result.n_treatment, 3)

    def test_permutation_mean_diff_has_effect_and_permutation_pvalue(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 1, 1, 1],
                "y": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
            }
        )
        result = PermutationMeanDiffEstimator(n_resamples=99, seed=4).estimate(
            data,
            ExperimentSpec(),
        )
        self.assertAlmostEqual(result.effect, 2.0)
        self.assertTrue(math.isfinite(result.standard_error))
        self.assertEqual(result.diagnostics["p_value_method"], "permutation_randomization")

    def test_bootstrap_mean_diff_is_reproducible(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 1, 1, 1],
                "y": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
            }
        )
        estimator = BootstrapMeanDiffEstimator(n_resamples=199, seed=8)
        first = estimator.estimate(data, ExperimentSpec())
        second = estimator.estimate(data, ExperimentSpec())
        self.assertAlmostEqual(first.effect, 2.0)
        self.assertEqual(first.to_record(), second.to_record())
        self.assertLessEqual(first.ci_low, first.effect)
        self.assertGreaterEqual(first.ci_high, first.effect)
        self.assertEqual(first.diagnostics["ci_method"], "studentized_bootstrap_t")
        self.assertEqual(first.diagnostics["p_value_method"], "studentized_bootstrap_t")

    def test_yuen_trimmed_estimator_matches_scipy(self) -> None:
        from scipy import stats as scipy_stats

        control = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        treatment = np.array([1.0, 2.0, 3.0, 4.0, 200.0])
        data = pd.DataFrame(
            {
                "treatment": np.r_[np.zeros(control.size), np.ones(treatment.size)],
                "y": np.r_[control, treatment],
            }
        )
        result = TrimmedMeanEstimator(trim_fraction=0.2).estimate(
            data,
            ExperimentSpec(),
        )
        reference = scipy_stats.ttest_ind(
            treatment,
            control,
            equal_var=False,
            trim=0.2,
        )
        self.assertAlmostEqual(result.effect, 1.0)
        self.assertAlmostEqual(result.statistic, float(reference.statistic), places=12)
        self.assertAlmostEqual(result.p_value, float(reference.pvalue), places=12)

    def test_population_stratification_uses_external_population_weights(self) -> None:
        rows = []
        for stratum, effect in (("low", 0.0), ("high", 4.0)):
            for treatment in (0, 1):
                for baseline in (0.0, 1.0, 2.0):
                    rows.append(
                        {
                            "stratum": stratum,
                            "treatment": treatment,
                            "y": baseline + treatment * effect,
                        }
                    )
        data = pd.DataFrame(rows)
        spec = ExperimentSpec(stratum_col="stratum")
        sample = StratifiedDifferenceInMeansEstimator().estimate(data, spec)
        population = PopulationWeightedStratifiedEstimator(
            population_weights={"low": 0.75, "high": 0.25}
        ).estimate(data, spec)
        self.assertAlmostEqual(sample.effect, 2.0)
        self.assertAlmostEqual(population.effect, 1.0)
        self.assertEqual(
            population.diagnostics["weight_source"],
            "known_target_population",
        )

    def test_entropy_calibration_targets_known_population_moment(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
                "population_score": [-1.0, -1.0, 0.0, 1.0] * 2,
                "y": [-2.0, -2.0, 0.0, 2.0, -1.0, -1.0, 1.0, 3.0],
            }
        )
        result = CalibrationWeightedATEEstimator(
            population_moments={"population_score": -0.5}
        ).estimate(data, ExperimentSpec())
        self.assertAlmostEqual(result.effect, 1.0, places=10)
        self.assertLess(result.diagnostics["max_balance_error"], 1e-7)

    def test_optimal_transport_matching_is_exact_and_capacity_bounded(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0, 0, 1, 1, 1],
                "y": [0.0, 10.0, 1.0, 2.0, 9.0],
                "x": [0.0, 10.0, 1.0, 2.0, 9.0],
            }
        )
        result = OptimalTransportMatchingEstimator().estimate(
            data,
            ExperimentSpec(pre_treatment_cols=("x",)),
        )
        self.assertEqual(result.diagnostics["control_reuse_capacity"], 2)
        self.assertEqual(result.diagnostics["matched_pairs"], 3)
        self.assertLessEqual(result.diagnostics["unique_controls"], 2)
        self.assertTrue(math.isfinite(result.effect))

    def test_cuped_wrapper_reduces_standard_error_on_correlated_synthetic_data(self) -> None:
        config = SyntheticConfig(n=6000, effect=0.2, covariate_signal=1.2, seed=7)
        data = generate_synthetic_ab(config)
        spec = ExperimentSpec(pre_target_col="y_pre")
        base = DifferenceInMeansEstimator()
        wrapped = WrappedEstimator(CupedTransformer(), DifferenceInMeansEstimator())
        raw = base.estimate(data, spec)
        adjusted = wrapped.estimate(data, spec)
        self.assertLess(adjusted.standard_error, raw.standard_error)

    def test_default_race_contains_hypex_baselines_and_runs(self) -> None:
        config = SyntheticConfig(n=2000, effect=0.25, seed=11)
        data = generate_synthetic_ab(config)
        historical = generate_historical_controls(config)
        spec = ExperimentSpec(
            covariate_cols=("y_pre", "x1", "x2"),
            pre_treatment_cols=("y_pre", "x1", "x2"),
            expert_safe_in_time_cols=("calendar_index", "market_index"),
            gated_in_time_cols=("operational_load",),
            pre_target_col="y_pre",
            numerator_col="numerator",
            denominator_col="denominator",
            stratum_col="stratum",
            instrument_col="assigned",
            received_treatment_col="received",
        )
        race = build_default_race(spec=spec)
        self.assertEqual(race[0].name, "welch_mean_diff")
        results = run_estimator_race(
            data,
            spec,
            race,
            auxiliary_data=historical,
        )
        names = set(results["estimator"])
        self.assertIn("hypex_ab", names)
        self.assertIn("hypex_cuped", names)
        self.assertIn("cuped_control__hypex_ab", names)
        self.assertIn("hypex_cupac_temporal", names)
        self.assertIn("cupac_linear_A__hypex_ab", names)
        self.assertIn("cupac_linear_A_plus_B__hypex_ab", names)
        self.assertIn("cupac_linear_A_plus_B_plus_C__hypex_ab", names)
        ok = results.set_index("estimator").loc[
            [
                "hypex_ab",
                "hypex_cuped",
                "cuped_control__hypex_ab",
                "hypex_cupac_temporal",
                "cupac_linear_A__hypex_ab",
                "cupac_linear_A_plus_B__hypex_ab",
                "cupac_linear_A_plus_B_plus_C__hypex_ab",
            ]
        ]
        self.assertTrue((ok["status"] == "ok").all())

    def test_default_race_contains_sheet_v02_methods(self) -> None:
        names = {estimator.name for estimator in build_default_race()}
        self.assertIn("ks_test", names)
        self.assertIn("approx_ks_fast", names)
        self.assertIn("energy_test", names)
        self.assertIn("randomized_t_test", names)
        self.assertIn("randomized_weighted_mean_diff", names)
        self.assertIn("regression_adjustment", names)
        self.assertIn("ml_rate", names)
        self.assertIn("nearest_neighbor_matching", names)
        self.assertIn("trimmed_mean_diff", names)
        self.assertIn("winsorized_mean_diff", names)
        self.assertIn("variance_weighted_stratified", names)
        self.assertIn("permutation_mean_test", names)

    def test_rnd6_cupac_feature_chain(self) -> None:
        spec = ExperimentSpec(
            pre_treatment_cols=("a1", "a2"),
            expert_safe_in_time_cols=("b1",),
            gated_in_time_cols=("c1",),
        )
        feature_sets = default_cupac_feature_sets(spec)
        self.assertEqual(feature_sets[0].method_id, "cupac_linear_A")
        self.assertEqual(feature_sets[0].zones, ("A",))
        self.assertEqual(feature_sets[1].zones, ("A", "B"))
        self.assertEqual(
            feature_sets[1].safety_status,
            "exploratory_expert_safe_in_time",
        )
        self.assertEqual(feature_sets[2].zones, ("A", "B", "C"))

    def test_zoned_cupac_needs_no_auxiliary_sample(self) -> None:
        data = generate_synthetic_ab(
            SyntheticConfig(n=1500, effect=0.2, seed=31)
        )
        spec = ExperimentSpec(
            pre_target_col="y_pre",
            pre_treatment_cols=("y_pre", "x1", "x2"),
            expert_safe_in_time_cols=("calendar_index", "market_index"),
            gated_in_time_cols=("operational_load",),
        )
        policies = default_cupac_feature_sets(spec)
        a = ZonedCupacTransformer(policies[0]).fit_transform(data, spec)
        ab_transformer = ZonedCupacTransformer(policies[1])
        ab = ab_transformer.fit_transform(data, spec)
        abc_transformer = ZonedCupacTransformer(
            replace(
                policies[2],
                max_abs_smd=10.0,
                max_missing_rate_difference=1.0,
                max_ks_distance=1.0,
            )
        )
        abc = abc_transformer.fit_transform(data, spec)
        self.assertEqual(len(a), len(data))
        self.assertEqual(len(ab), len(data))
        self.assertEqual(len(abc), len(data))
        self.assertEqual(ab_transformer.diagnostics()["stage_count"], 2)
        self.assertEqual(
            ab_transformer.diagnostics()["stage_1"],
            "cupac_A",
        )
        self.assertEqual(
            ab_transformer.diagnostics()["stage_2"],
            "linear_adjustment_B",
        )
        self.assertEqual(abc_transformer.diagnostics()["stage_count"], 3)
        self.assertEqual(
            abc_transformer.diagnostics()["stage_3"],
            "linear_adjustment_C",
        )

    def test_temporal_cupac_tunes_ridge_on_historical_target_only(self) -> None:
        data = generate_synthetic_ab(SyntheticConfig(n=500, effect=0.2, seed=41))
        spec = ExperimentSpec(
            pre_target_col="y_pre",
            pre_treatment_cols=("y_pre", "x1", "x2"),
        )
        transformer = LinearCupacTransformer()
        adjusted = transformer.fit_transform(data, spec)
        self.assertEqual(len(adjusted), len(data))
        diagnostics = transformer.diagnostics()
        self.assertTrue(diagnostics["ridge_auto_tuned"])
        self.assertGreater(diagnostics["ridge_validation_candidates"], 0)
        self.assertFalse(diagnostics["current_outcome_used_for_model_fit"])

    def test_isolation_forest_transformer_is_deterministic_and_treatment_blind(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0] * 30 + [1] * 30,
                "y": [float(value) for value in range(30)]
                + [float(value) for value in range(30)],
            }
        )
        data.loc[0, "y"] = 1000.0
        spec = ExperimentSpec()
        first = IsolationForestTargetTransformer(
            method="winsorize", contamination=0.05, random_state=9
        )
        second = IsolationForestTargetTransformer(
            method="winsorize", contamination=0.05, random_state=9
        )
        first_values = first.fit_transform(data, spec)
        second_values = second.fit_transform(data, spec)
        np.testing.assert_allclose(first_values, second_values, equal_nan=True)
        self.assertEqual(
            first.diagnostics()["threshold_selection"],
            "pooled_score_quantile_without_group_labels",
        )
        self.assertGreaterEqual(first.diagnostics()["outlier_count"], 1)

    def test_zone_c_gate_rejects_treatment_dependent_feature(self) -> None:
        data = generate_synthetic_ab(
            SyntheticConfig(
                n=3000,
                effect=0.0,
                gated_treatment_shift=0.8,
                seed=37,
            )
        )
        spec = ExperimentSpec(
            pre_target_col="y_pre",
            pre_treatment_cols=("y_pre", "x1", "x2"),
            expert_safe_in_time_cols=("calendar_index", "market_index"),
            gated_in_time_cols=("operational_load",),
        )
        transformer = ZonedCupacTransformer(default_cupac_feature_sets(spec)[2])
        transformer.fit_transform(data, spec)
        self.assertEqual(
            transformer.diagnostics()["zone_c_rejected"],
            "operational_load",
        )
        self.assertEqual(transformer.diagnostics()["stage_count"], 2)

    def test_ratio_delta_returns_finite_estimate(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0, 0, 1, 1],
                "numerator": [10.0, 12.0, 14.0, 18.0],
                "denominator": [5.0, 6.0, 7.0, 9.0],
            }
        )
        spec = ExperimentSpec(numerator_col="numerator", denominator_col="denominator")
        result = RatioDeltaMethodEstimator().estimate(data, spec)
        self.assertTrue(math.isfinite(result.effect))

    def test_randomized_t_test_is_finite(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
                "y": [1.0, 1.2, 0.9, 1.1, 1.4, 1.6, 1.3, 1.5],
            }
        )
        result = RandomizedTTestEstimator().estimate(data, ExperimentSpec())
        self.assertTrue(math.isfinite(result.statistic))
        self.assertTrue(0.0 <= result.p_value <= 1.0)

    def test_randomized_weighted_mean_diff_has_coherent_inference(self) -> None:
        data = pd.DataFrame(
            {
                "unit": np.arange(80),
                "treatment": np.tile([0, 1], 40),
                "y": np.linspace(-1.0, 3.0, 80),
            }
        )
        spec = ExperimentSpec(unit_col="unit")
        result = RandomizedWeightedMeanDiffEstimator(seed=17).estimate(data, spec)
        self.assertTrue(math.isfinite(result.effect))
        self.assertTrue(math.isfinite(result.standard_error))
        self.assertTrue(math.isfinite(result.p_value))
        self.assertTrue(math.isfinite(result.ci_low))
        self.assertTrue(math.isfinite(result.ci_high))
        self.assertEqual(
            result.diagnostics["estimation_target"],
            "difference_of_normalized_hedgehog_weighted_means",
        )

    def test_randomized_weighted_mean_diff_is_reproducible_and_order_safe(self) -> None:
        rng = np.random.default_rng(91)
        data = pd.DataFrame(
            {
                "unit": np.arange(120),
                "treatment": np.tile([0, 1], 60),
                "y": rng.normal(size=120),
            }
        )
        spec = ExperimentSpec(unit_col="unit")
        estimator = RandomizedWeightedMeanDiffEstimator(seed=23)
        original = estimator.estimate(data, spec)
        reordered = estimator.estimate(
            data.sample(frac=1.0, random_state=123),
            spec,
        )
        repeated = estimator.estimate(data, spec)
        self.assertAlmostEqual(original.effect, reordered.effect, places=12)
        self.assertAlmostEqual(original.standard_error, reordered.standard_error, places=12)
        self.assertAlmostEqual(original.p_value, reordered.p_value, places=12)
        self.assertAlmostEqual(original.effect, repeated.effect, places=12)

    def test_randomized_weighted_mean_diff_is_mean_unbiased_in_simulation(self) -> None:
        rng = np.random.default_rng(1234)
        effects = []
        estimator = RandomizedWeightedMeanDiffEstimator(seed=20260731)
        for _ in range(80):
            control = rng.normal(0.0, 1.0, 400)
            treatment = rng.normal(0.35, 1.0, 400)
            data = pd.DataFrame(
                {
                    "treatment": np.r_[np.zeros(400), np.ones(400)],
                    "y": np.r_[control, treatment],
                }
            )
            effects.append(estimator.estimate(data, ExperimentSpec()).effect)
        self.assertLess(abs(float(np.mean(effects)) - 0.35), 0.08)

    def test_approx_ks_tracks_exact_ks_on_synthetic_data(self) -> None:
        data = generate_synthetic_ab(SyntheticConfig(n=1000, effect=0.15, seed=17))
        exact = KSTestEstimator().estimate(data, ExperimentSpec())
        approx = ApproxKSTestEstimator(n_bins=256).estimate(data, ExperimentSpec())
        self.assertLess(abs(exact.effect - approx.effect), 0.03)
        self.assertTrue(0.0 <= approx.p_value <= 1.0)

    def test_energy_test_is_finite(self) -> None:
        data = generate_synthetic_ab(SyntheticConfig(n=500, effect=0.30, seed=19))
        result = EnergyDistanceTestEstimator(n_resamples=29).estimate(data, ExperimentSpec())
        self.assertTrue(math.isfinite(result.effect))
        self.assertTrue(0.0 <= result.p_value <= 1.0)

    def test_ml_rate_and_matching_are_finite(self) -> None:
        data = generate_synthetic_ab(SyntheticConfig(n=1200, effect=0.20, seed=23))
        spec = ExperimentSpec(
            covariate_cols=("y_pre", "x1", "x2"),
            pre_treatment_cols=("y_pre", "x1", "x2"),
        )
        ml_rate = MLRateEstimator(n_splits=3).estimate(data, spec)
        matching = NearestNeighborMatchingEstimator().estimate(data, spec)
        self.assertTrue(math.isfinite(ml_rate.effect))
        self.assertTrue(math.isfinite(ml_rate.standard_error))
        self.assertTrue(math.isfinite(matching.effect))
        self.assertTrue(math.isfinite(matching.standard_error))

    def test_hypex_backend_hook_can_delegate(self) -> None:
        def backend(data: pd.DataFrame, spec: ExperimentSpec):
            result = DifferenceInMeansEstimator().estimate(data, spec)
            return result

        data = pd.DataFrame(
            {"treatment": [0, 0, 1, 1], "y": [1.0, 2.0, 4.0, 5.0]}
        )
        result = HypExABTestEstimator(backend=backend).estimate(data, ExperimentSpec())
        self.assertEqual(result.estimator, "hypex_ab")
        self.assertEqual(result.diagnostics["backend"], "external_hypex")

    def test_pypi_hypex_cupac_uses_adjusted_target_row(self) -> None:
        try:
            import hypex  # noqa: F401
        except ImportError:
            self.skipTest("PyPI HypEx is not installed in this runtime")

        config = SyntheticConfig(n=1200, effect=0.25, seed=101)
        data = generate_synthetic_ab(config)
        spec = ExperimentSpec(
            pre_target_col="y_pre",
            pre_treatment_cols=("y_pre", "x1", "x2"),
        )
        raw = HypExABTestEstimator(prefer_pypi=True).estimate(data, spec)
        adjusted = HypExCupacEstimator(
            cupac_features=("y_pre", "x1", "x2"),
            prefer_pypi=True,
        ).estimate(data, spec)

        self.assertEqual(adjusted.diagnostics["backend"], "pypi_hypex")
        self.assertEqual(adjusted.diagnostics["variance_reduction"], "cupac")
        self.assertNotAlmostEqual(adjusted.effect, raw.effect, places=10)
        self.assertLess(adjusted.standard_error, raw.standard_error)

    def test_synthetic_noise_families_are_finite_and_reproducible(self) -> None:
        for distribution in (
            "normal",
            "student_t",
            "centered_lognormal",
            "contaminated_normal",
        ):
            config = SyntheticConfig(
                n=500,
                seed=701,
                noise_distribution=distribution,
                experiment_stratum_weights=(0.2, 0.4, 0.4),
            )
            first = generate_synthetic_ab(config)
            second = generate_synthetic_ab(config)
            self.assertTrue(np.isfinite(first["y"]).all())
            self.assertTrue(first.equals(second))
            self.assertIn("population_stratum", first)

    def test_pypi_hypex_ab_ignores_irrelevant_categorical_columns(self) -> None:
        try:
            import hypex  # noqa: F401
        except ImportError:
            self.skipTest("PyPI HypEx is not installed in this runtime")

        data = generate_synthetic_ab(SyntheticConfig(n=600, effect=0.20, seed=103))
        spec = ExperimentSpec(
            covariate_cols=("y_pre", "region", "device"),
            pre_treatment_cols=("y_pre", "region", "device"),
        )
        result = HypExABTestEstimator(
            prefer_pypi=True,
            require_pypi=True,
        ).estimate(data, spec)

        self.assertEqual(result.diagnostics["backend"], "pypi_hypex")
        self.assertEqual(
            result.diagnostics["standard_error_source"],
            "welch_on_hypex_output_target",
        )

    def test_placeholder_is_explicit(self) -> None:
        placeholder = PlaceholderEstimator("unknown", "needs research")
        with self.assertRaises(NotImplementedError):
            placeholder.estimate(pd.DataFrame({"treatment": [], "y": []}), ExperimentSpec())

    def test_multiple_testing_helpers(self) -> None:
        adjusted = bonferroni([0.01, 0.20, 0.60])
        np.testing.assert_allclose(adjusted, [0.03, 0.60, 1.00])
        holm = holm_bonferroni([0.01, 0.04, 0.03])
        self.assertTrue(np.all(np.diff(np.sort(holm)) >= 0))
        observed = np.array([3.0, 2.0, 1.0])
        bootstrap = np.array(
            [
                [2.5, 1.5, 0.5],
                [1.0, 2.5, 1.2],
                [0.5, 0.7, 1.5],
            ]
        )
        romano = romano_wolf_stepdown(observed, bootstrap)
        self.assertEqual(romano.shape, observed.shape)
        self.assertTrue(np.all((0.0 <= romano) & (romano <= 1.0)))


if __name__ == "__main__":
    unittest.main()
