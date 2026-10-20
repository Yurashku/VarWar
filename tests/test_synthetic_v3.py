from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dataclasses import replace

from varwar.synthetic_v3 import (
    WORLD_CATALOG,
    generate_world,
    get_world_spec,
    spec_for_estimand,
    world_covariate_cols,
    world_noise_cols,
    world_pre_treatment_cols,
)


class SyntheticV3Tests(unittest.TestCase):
    def test_exact_ground_truth_comes_from_potential_outcomes(self) -> None:
        world = generate_world(
            get_world_spec("population_shift", n_experiment=300, population_size=1_200),
            seed=17,
        )
        expected_sample = float(
            np.mean(world.potential_outcomes["y1"] - world.potential_outcomes["y0"])
        )
        expected_population = float(
            np.mean(world.finite_population["y1"] - world.finite_population["y0"])
        )
        self.assertAlmostEqual(world.true_sample_ate, expected_sample)
        self.assertAlmostEqual(world.true_population_ate, expected_population)
        self.assertNotAlmostEqual(world.true_sample_ate, world.true_population_ate, places=2)
        self.assertEqual(
            {"trimmed_mean_10", "winsorized_mean_10", "median_effect"},
            set(world.native_truths),
        )

    def test_pre_period_panel_has_stable_entities(self) -> None:
        spec = get_world_spec(
            "gaussian_r2_04_n200",
            population_size=500,
            n_pre_periods=5,
        )
        world = generate_world(spec, seed=21)
        counts = world.pre_period_panel.groupby("entity_id", observed=True).size()
        self.assertEqual(len(counts), spec.population_size)
        self.assertTrue((counts == spec.n_pre_periods).all())
        self.assertEqual(spec.n_pre_periods, world.pre_period_panel["report_date"].nunique())

    def test_assignment_data_has_no_assignment_or_outcomes(self) -> None:
        world = generate_world(
            get_world_spec("temporal_c_unsafe", n_experiment=200, population_size=600),
            seed=23,
        )
        forbidden_fragments = (
            "treatment",
            "assigned",
            "received",
            "outcome",
            "y0",
            "y1",
            "effect",
            "zone_b",
            "zone_c",
            "post_",
        )
        for column in world.assignment_data.columns:
            self.assertFalse(
                any(fragment in column.lower() for fragment in forbidden_fragments),
                column,
            )
        self.assertTrue({"pre_target_lag1", "pre_target_lag2"}.issubset(world.assignment_data))
        self.assertEqual(len(world.assignment_data), world.spec.n_experiment)
        self.assertTrue(world.assignment_data["entity_id"].is_unique)

    def test_world_is_deterministic_for_fixed_seed(self) -> None:
        spec = get_world_spec(
            "interactions_high_dimensional",
            n_experiment=120,
            population_size=400,
        )
        first = generate_world(spec, seed=29)
        second = generate_world(spec, seed=29)
        pd.testing.assert_frame_equal(first.finite_population, second.finite_population)
        pd.testing.assert_frame_equal(first.pre_period_panel, second.pre_period_panel)
        pd.testing.assert_frame_equal(first.assignment_data, second.assignment_data)
        self.assertEqual(first.true_sample_ate, second.true_sample_ate)
        third = generate_world(spec, seed=30)
        self.assertFalse(first.finite_population.equals(third.finite_population))

    def test_catalog_covers_required_world_dimensions(self) -> None:
        specs = tuple(WORLD_CATALOG.values())
        self.assertTrue({0.0, 0.4, 0.8}.issubset({spec.r2 for spec in specs}))
        self.assertTrue({200, 2_000}.issubset({spec.n_experiment for spec in specs}))
        self.assertTrue({0.3, 0.5}.issubset({spec.treatment_share for spec in specs}))
        self.assertTrue(
            {"gaussian", "student_t3", "lognormal", "contaminated", "zero_inflated_count"}
            .issubset({spec.outcome_distribution for spec in specs})
        )
        self.assertTrue({"linear", "quadratic", "threshold", "interactions"}.issubset({spec.signal_model for spec in specs}))
        self.assertTrue(any(spec.heteroskedastic for spec in specs))
        self.assertTrue(any(spec.heterogeneous_effect for spec in specs))
        self.assertTrue(any(spec.many_rare_strata for spec in specs))
        self.assertTrue(any(spec.panel_shift for spec in specs))
        self.assertTrue(any(spec.temporal_drift for spec in specs))
        self.assertTrue(any(spec.prior_mismatch for spec in specs))


class EffectModifierCenteringTests(unittest.TestCase):
    def test_default_centering_keeps_population_truth_exact(self) -> None:
        spec = get_world_spec("population_shift", n_experiment=300, population_size=1_500)
        self.assertEqual(spec.effect_modifier_centering, "population")
        world = generate_world(replace(spec, effect=0.0), seed=3)
        self.assertLess(abs(world.true_population_ate), 1e-12)
        self.assertGreater(abs(world.true_sample_ate), 0.1)

    def test_cohort_centering_gives_true_sample_null(self) -> None:
        for world_id in ("population_shift", "heterogeneous_effect", "population_shift_panel"):
            base = get_world_spec(world_id, n_experiment=300, population_size=1_500, effect=0.0)
            sample_spec = spec_for_estimand(base, "sample_ate")
            self.assertEqual(sample_spec.effect_modifier_centering, "cohort")
            for seed in (1, 2, 3):
                world = generate_world(sample_spec, seed=seed)
                self.assertLess(abs(world.true_sample_ate), 1e-12, world_id)
                self.assertTrue(
                    np.allclose(
                        world.potential_outcomes["y1"] - world.potential_outcomes["y0"],
                        world.potential_outcomes["individual_effect"],
                    )
                )

    def test_both_centerings_share_draws_and_record_both_truths(self) -> None:
        base = get_world_spec("population_shift", n_experiment=300, population_size=1_500)
        population = generate_world(base, seed=9)
        cohort = generate_world(replace(base, effect_modifier_centering="cohort"), seed=9)
        pd.testing.assert_frame_equal(population.assignment_data, cohort.assignment_data)
        pd.testing.assert_series_equal(
            population.potential_outcomes["y0"], cohort.potential_outcomes["y0"]
        )
        shift = population.true_sample_ate - cohort.true_sample_ate
        self.assertGreater(shift, 0.1)
        self.assertAlmostEqual(
            cohort.true_population_ate, population.true_population_ate - shift, places=12
        )
        for world in (population, cohort):
            self.assertAlmostEqual(
                world.metadata["truths_population_centering"]["sample_ate"],
                population.true_sample_ate,
                places=12,
            )
            self.assertAlmostEqual(
                world.metadata["truths_cohort_centering"]["population_ate"],
                cohort.true_population_ate,
                places=12,
            )
            self.assertEqual(
                world.estimand_truths["sample_ate"], world.true_sample_ate
            )

    def test_spec_for_estimand_leaves_homogeneous_worlds_alone(self) -> None:
        spec = get_world_spec("gaussian_r2_04_n200")
        self.assertIs(spec_for_estimand(spec, "sample_ate"), spec)
        hetero = get_world_spec("heterogeneous_effect", effect_modifier_centering="cohort")
        self.assertEqual(
            spec_for_estimand(hetero, "population_ate").effect_modifier_centering,
            "population",
        )
        self.assertIs(spec_for_estimand(hetero, "trimmed_mean_effect"), hetero)

    def test_invalid_centering_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_world_spec("heterogeneous_effect", effect_modifier_centering="sample")


class PanelWorldTests(unittest.TestCase):
    def test_outcome_continues_the_pre_period_series(self) -> None:
        world = generate_world(get_world_spec("gaussian_n2000_panel"), seed=4)
        metadata = world.metadata
        self.assertEqual(metadata["outcome_process"], "panel")
        # Y_it = alpha_i + f(X_i) + d_t + eps_it: corr(y, pre) = r2 + unit share.
        self.assertGreater(metadata["oracle_corr2_y_pre"], 0.3)
        self.assertAlmostEqual(metadata["oracle_r2"], 0.6, delta=0.05)
        self.assertAlmostEqual(metadata["oracle_x_r2"], 0.2, delta=0.03)
        y0 = world.potential_outcomes["y0"].to_numpy()
        lag1 = world.assignment_data["pre_target_lag1"].to_numpy()
        lag2 = world.assignment_data["pre_target_lag2"].to_numpy()
        self.assertAlmostEqual(np.corrcoef(y0, lag1)[0, 1], np.corrcoef(lag1, lag2)[0, 1], delta=0.06)
        static = generate_world(get_world_spec("gaussian_r2_04_n2000"), seed=4)
        self.assertLess(static.metadata["oracle_corr2_y_pre"], 0.25)

    def test_external_panel_excludes_cohort_and_carries_the_shift(self) -> None:
        clean = generate_world(get_world_spec("gaussian_n2000_panel"), seed=6)
        shifted = generate_world(get_world_spec("external_prior_mismatch_panel"), seed=6)
        cohort = set(clean.assignment_data["entity_id"])
        self.assertFalse(cohort.intersection(clean.pre_period_panel["entity_id"]))
        self.assertEqual(
            clean.pre_period_panel["entity_id"].nunique(),
            clean.spec.population_size - clean.spec.n_experiment,
        )
        # The shift lives only in the external aggregates.
        pd.testing.assert_frame_equal(clean.assignment_data, shifted.assignment_data)
        pd.testing.assert_frame_equal(clean.potential_outcomes, shifted.potential_outcomes)
        difference = (
            shifted.pre_period_panel["pre_target"] - clean.pre_period_panel["pre_target"]
        ).abs()
        self.assertGreater(float(difference.mean()), 0.5)

    def test_panel_world_validation(self) -> None:
        with self.assertRaises(ValueError):
            get_world_spec("gaussian_n2000_panel", unit_effect_share=0.9)
        with self.assertRaises(ValueError):
            get_world_spec("gaussian_r2_04_n200", unit_effect_share=0.2)
        with self.assertRaises(ValueError):
            get_world_spec("gaussian_n2000_panel", temporal_drift=0.5)

    def test_freedman_world_has_unequal_split_and_prognostic_heterogeneity(self) -> None:
        for world_id, share in (("freedman_p20_panel", 0.2), ("freedman_p30_panel", 0.3)):
            spec = get_world_spec(world_id, n_experiment=800, population_size=4_000)
            self.assertEqual(spec.treatment_share, share)
            world = generate_world(spec_for_estimand(replace(spec, effect=0.0), "sample_ate"), seed=2)
            outcomes = world.potential_outcomes
            lag1 = world.assignment_data["pre_target_lag1"].to_numpy()
            effect = outcomes["individual_effect"].to_numpy()
            # The effect modifier is the persistent prognostic component with a
            # sign opposite to the control slope, and the arms differ in variance.
            self.assertLess(np.corrcoef(effect, lag1)[0, 1], -0.5)
            self.assertGreater(outcomes["y1"].var(), 1.0)
            self.assertLess(abs(world.true_sample_ate), 1e-12)

    def test_nonlinear_panel_has_headroom_over_linear_features(self) -> None:
        world = generate_world(get_world_spec("nonlinear_panel"), seed=5)
        self.assertGreater(
            world.metadata["oracle_r2"],
            world.metadata["linear_r2_pre_treatment_features"] + 0.1,
        )


class HighDimensionalCovariateTests(unittest.TestCase):
    def test_noise_columns_are_declared_and_lag_aligned(self) -> None:
        spec = get_world_spec("high_dimensional_covariates_n500")
        columns = world_pre_treatment_cols(spec)
        self.assertEqual(len(world_noise_cols(spec)), 24)
        self.assertTrue(set(world_noise_cols(spec)).issubset(columns))
        self.assertTrue(set(world_noise_cols(spec)).issubset(world_covariate_cols(spec)))
        world = generate_world(spec, seed=8)
        for column in world_noise_cols(spec):
            self.assertIn(column, world.assignment_data)
            np.testing.assert_array_equal(
                world.assignment_data[column], world.assignment_data[f"{column}__lag2"]
            )
        default = get_world_spec("interactions_high_dimensional")
        self.assertEqual(world_pre_treatment_cols(default), ("pre_target_lag1", "x1", "x2", "x3", "x4"))

    def test_catalog_contains_v31_worlds(self) -> None:
        expected = {
            "gaussian_r2_04_n500",
            "heteroskedastic_n500",
            "high_dimensional_covariates_n500",
            "gaussian_n200_panel",
            "gaussian_n500_panel",
            "gaussian_n2000_panel",
            "heterogeneous_effect_panel",
            "population_shift_panel",
            "external_shift_panel",
            "external_prior_mismatch_panel",
            "nonlinear_panel",
            "freedman_p20_panel",
            "freedman_p30_panel",
        }
        self.assertTrue(expected.issubset(WORLD_CATALOG))
        self.assertIn(500, {spec.n_experiment for spec in WORLD_CATALOG.values()})
        self.assertTrue({0.2, 0.3}.issubset({spec.treatment_share for spec in WORLD_CATALOG.values()}))


if __name__ == "__main__":
    unittest.main()
