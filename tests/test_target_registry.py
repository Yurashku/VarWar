from __future__ import annotations

import unittest

from varwar.analysis import CausalSafety, InformationMode, TemporalAccess
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

    def test_registry_rebuild_keeps_stable_ids(self) -> None:
        rebuilt = build_target_registry()
        self.assertEqual(tuple(rebuilt), tuple(TARGET_REGISTRY))
        self.assertIn("identity", rebuilt)
        self.assertIn("cuped_historical", rebuilt)
        self.assertIn("cuped_huber_historical", rebuilt)


if __name__ == "__main__":
    unittest.main()
