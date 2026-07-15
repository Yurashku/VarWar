from __future__ import annotations

import math
import sys
import unittest
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
    DifferenceInMeansEstimator,
    EnergyDistanceTestEstimator,
    HypExABTestEstimator,
    KSTestEstimator,
    MLRateEstimator,
    NearestNeighborMatchingEstimator,
    PlaceholderEstimator,
    RandomizedTTestEstimator,
    RatioDeltaMethodEstimator,
)
from varwar.synthetic import SyntheticConfig, generate_historical_controls, generate_synthetic_ab
from varwar.feature_sets import default_cupac_feature_sets
from varwar.transforms import CupedTransformer


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
        self.assertIn("cuped+hypex_ab", names)
        self.assertIn("hypex_cupac", names)
        self.assertIn("sklearn_cupac_A+hypex_ab", names)
        self.assertIn("sklearn_cupac_A_plus_B+hypex_ab", names)
        self.assertIn("sklearn_cupac_A_plus_B_plus_C+hypex_ab", names)
        ok = results.set_index("estimator").loc[
            [
                "hypex_ab",
                "cuped+hypex_ab",
                "hypex_cupac",
                "sklearn_cupac_A+hypex_ab",
                "sklearn_cupac_A_plus_B+hypex_ab",
                "sklearn_cupac_A_plus_B_plus_C+hypex_ab",
            ]
        ]
        self.assertTrue((ok["status"] == "ok").all())

    def test_default_race_contains_sheet_v02_methods(self) -> None:
        names = {estimator.name for estimator in build_default_race()}
        self.assertIn("ks_test", names)
        self.assertIn("approx_ks_fast", names)
        self.assertIn("energy_test", names)
        self.assertIn("randomized_t_test", names)
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
        self.assertEqual(feature_sets[0].name, "sklearn_cupac_A")
        self.assertEqual(feature_sets[0].columns, ("a1", "a2"))
        self.assertEqual(feature_sets[1].columns, ("a1", "a2", "b1"))
        self.assertEqual(feature_sets[2].columns, ("a1", "a2", "b1", "c1"))

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
