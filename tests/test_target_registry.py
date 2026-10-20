from __future__ import annotations

import unittest

import numpy as np

from varwar.analysis import CausalSafety, InformationMode, TemporalAccess
from varwar.core import ExperimentSpec
from varwar.synthetic_v3 import generate_world
from varwar.target_registry import (
    TARGET_REGISTRY,
    TARGET_REGISTRY_METADATA,
    build_target_registry,
)


class TargetRegistryTests(unittest.TestCase):
    def test_every_cupac_family_has_all_three_zone_variants(self) -> None:
        families = (
            "linear",
            "quadratic",
            "automl",
            "hypex_compatible",
            "hypex_official_target",
        )
        suffixes = ("A", "A_plus_B", "A_plus_B_plus_C")
        for family in families:
            for suffix in suffixes:
                method_id = f"cupac_{family}_{suffix}"
                self.assertIn(method_id, TARGET_REGISTRY)
                self.assertEqual(
                    TARGET_REGISTRY[method_id].metadata["zones"],
                    tuple(suffix.split("_plus_")),
                )
        for suffix in suffixes:
            method_id = f"panel_prior_cupac_{suffix}"
            self.assertIn(method_id, TARGET_REGISTRY)
            self.assertEqual(
                TARGET_REGISTRY[method_id].metadata["zones"],
                tuple(suffix.split("_plus_")),
            )

    def test_zone_contracts_are_explicit_and_not_separate_contours(self) -> None:
        a = TARGET_REGISTRY["cupac_automl_A"]
        ab = TARGET_REGISTRY["cupac_automl_A_plus_B"]
        abc = TARGET_REGISTRY["cupac_automl_A_plus_B_plus_C"]
        self.assertEqual(a.minimum_temporal_access, TemporalAccess.PRE_TREATMENT_ONLY)
        self.assertEqual(ab.minimum_temporal_access, TemporalAccess.FULL_TIMELINE)
        self.assertEqual(abc.minimum_temporal_access, TemporalAccess.FULL_TIMELINE)
        self.assertEqual(ab.causal_safety, CausalSafety.CONDITIONAL)
        self.assertEqual(abc.causal_safety, CausalSafety.POST_TREATMENT_RISK)
        self.assertTrue(ab.metadata["b_c_are_separate_control_fitted_stages"])
        self.assertEqual(ab.information_modes, frozenset(InformationMode))

    def test_panel_prior_is_the_only_restricted_available_family(self) -> None:
        panel = TARGET_REGISTRY["panel_prior_cupac_A"]
        linear = TARGET_REGISTRY["cupac_linear_A"]
        self.assertEqual(panel.information_modes, frozenset({InformationMode.PANEL_PRIOR}))
        self.assertEqual(linear.information_modes, frozenset(InformationMode))
        self.assertIn("panel_prior_summary", panel.required_columns)

    def test_factories_return_fresh_objects_and_fresh_nested_stages(self) -> None:
        for method_id, definition in TARGET_REGISTRY.items():
            if not definition.available:
                continue
            first = definition.create()
            second = definition.create()
            self.assertIsNot(first, second, method_id)
            if method_id.startswith("cupac_") and "_A_plus_" in method_id:
                self.assertIsNot(first.a_factory(), second.a_factory(), method_id)

    def test_unavailable_official_hypex_target_api_is_honest(self) -> None:
        definition = TARGET_REGISTRY["cupac_hypex_official_target_A"]
        self.assertFalse(definition.available)
        self.assertIn("no verified public HypEx target-only API", definition.unavailable_reason)
        with self.assertRaisesRegex(RuntimeError, "target-only API"):
            definition.create()

    def test_trial_lin_cuped_is_explicitly_excluded_from_target_layer(self) -> None:
        self.assertNotIn("cuped_trial_lin", TARGET_REGISTRY)
        reason = TARGET_REGISTRY_METADATA["excluded_from_target_layer"]["cuped_trial_lin"]
        self.assertIn("AnalysisMethod", reason)
        self.assertIn("HC2/HC3", reason)

    def test_pooled_theta_targets_are_registered(self) -> None:
        cuped = TARGET_REGISTRY["cuped_pooled_theta"]
        cupac = TARGET_REGISTRY["cupac_linear_pooled_theta_A"]
        for definition in (cuped, cupac):
            self.assertTrue(definition.available)
            self.assertEqual(definition.minimum_temporal_access, TemporalAccess.PRE_TREATMENT_ONLY)
            self.assertEqual(definition.causal_safety, CausalSafety.CONDITIONAL)
            self.assertTrue(definition.metadata["trial_outcome_used_for_theta"])
        self.assertEqual(cuped.create().fit_data, "experiment_all")
        self.assertEqual(cupac.create().theta_policy, "pooled")

    def test_pooled_theta_never_inflates_variance_where_fixed_coefficient_does(self) -> None:
        # strong_prior_mismatch: the lag2 -> lag1 relation does not transfer to
        # the current outcome, so the fixed coefficient of CUPAC A (and the AR
        # slope of historical CUPED) inflate variance; pooled theta does not.
        spec = ExperimentSpec(
            unit_col="entity_id",
            covariate_cols=("x1", "x2", "x3", "x4"),
            pre_treatment_cols=("pre_target_lag1", "x1", "x2", "x3", "x4"),
            pre_target_col="pre_target_lag1",
        )
        for seed in (1, 2):
            world = generate_world("strong_prior_mismatch", seed=seed)
            frame = world.assignment_data.copy()
            rng = np.random.default_rng(seed)
            treatment = np.zeros(len(frame), dtype=int)
            treatment[rng.choice(len(frame), size=len(frame) // 2, replace=False)] = 1
            frame["treatment"] = treatment
            frame["y"] = np.where(
                treatment == 1, world.experiment_cohort["y1"], world.experiment_cohort["y0"]
            )
            frame["pre_target_lag1__lag2"] = frame["pre_target_lag2"]
            for column in ("x1", "x2", "x3", "x4"):
                frame[f"{column}__lag2"] = frame[column]
            ratios = {}
            for method_id in (
                "cuped_pooled_theta",
                "cupac_linear_pooled_theta_A",
                "cupac_hypex_compatible_A",
                "cupac_linear_A",
                "cuped_historical",
            ):
                transformer = TARGET_REGISTRY[method_id].create()
                transformer.fit_transform(frame, spec)
                diagnostics = dict(transformer.diagnostics())
                ratios[method_id] = next(
                    value
                    for key, value in diagnostics.items()
                    if str(key).endswith("variance_ratio_within_arm")
                )
            self.assertLessEqual(ratios["cuped_pooled_theta"], 1.05)
            self.assertLessEqual(ratios["cupac_linear_pooled_theta_A"], 1.05)
            self.assertLessEqual(ratios["cupac_hypex_compatible_A"], 1.05)
            self.assertGreater(ratios["cupac_linear_A"], 1.5)
            self.assertGreater(ratios["cuped_historical"], 1.3)

    def test_registry_rebuild_keeps_stable_ids(self) -> None:
        rebuilt = build_target_registry()
        self.assertEqual(tuple(rebuilt), tuple(TARGET_REGISTRY))
        self.assertIn("identity", rebuilt)
        self.assertIn("cuped_historical", rebuilt)
        self.assertIn("cuped_huber_historical", rebuilt)


if __name__ == "__main__":
    unittest.main()
