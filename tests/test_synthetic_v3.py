from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from varwar.synthetic_v3 import WORLD_CATALOG, generate_world, get_world_spec


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


if __name__ == "__main__":
    unittest.main()
