from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from varwar.assignment import (
    AssignmentReplayTrace,
    CompleteRandomization,
    HypExAABestOfK,
    RerandomizationMahalanobis,
    RerandomizationMaxSMD,
    RerandomizationMixedBalance,
    StratifiedRandomization,
)
from varwar.core import ExperimentSpec


def _design_data(n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(18)
    index = pd.Index(np.arange(1000, 1000 + n_rows), name="row")
    x = rng.normal(size=n_rows)
    x[::17] = np.nan
    return pd.DataFrame(
        {
            "unit": [f"u-{index_value}" for index_value in index],
            "x": x,
            "z": rng.normal(size=n_rows),
            "count_with_ties": rng.integers(0, 5, size=n_rows),
            "category": np.resize(np.array(["a", "b", "c"], dtype=object), n_rows),
            "stratum": np.resize(np.array(["s1", "s2", "s3", "s4"], dtype=object), n_rows),
            "y": rng.normal(size=n_rows),
        },
        index=index,
    )


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        unit_col="unit",
        pre_treatment_cols=("x", "z", "count_with_ties", "category"),
        stratum_col="stratum",
    )


class BasicAssignmentTests(unittest.TestCase):
    def test_complete_randomization_is_exact_and_reproducible(self) -> None:
        data = _design_data(101)
        mechanism = CompleteRandomization(treatment_share=0.3)
        first = mechanism.assign(data, _spec(), np.random.default_rng(42))
        second = mechanism.assign(data, _spec(), np.random.default_rng(42))

        self.assertEqual(first.n_treatment, 30)
        self.assertEqual(first.n_control, 71)
        self.assertEqual(int((first.labels == 1).sum()), 30)
        self.assertTrue(first.labels.index.equals(data.index))
        pd.testing.assert_series_equal(first.labels, second.labels)
        self.assertIsInstance(first.replay_trace, AssignmentReplayTrace)
        self.assertIn("assignment_seed", first.replay_trace.seeds)

    def test_stratified_randomization_has_exact_total_and_near_exact_strata(self) -> None:
        data = _design_data(103)
        result = StratifiedRandomization(treatment_share=0.4).assign(
            data, _spec(), np.random.default_rng(7)
        )

        self.assertEqual(result.n_treatment, round(103 * 0.4))
        assigned = data.assign(treatment=result.labels)
        for _, group in assigned.groupby("stratum"):
            expected = len(group) * 0.4
            self.assertLessEqual(abs(group["treatment"].sum() - expected), 1.0)
            self.assertGreater(group["treatment"].sum(), 0)
            self.assertLess(group["treatment"].sum(), len(group))

    def test_stratified_randomization_rejects_singleton_strata(self) -> None:
        data = _design_data(20)
        data.loc[data.index[0], "stratum"] = "singleton"
        with self.assertRaisesRegex(ValueError, "at least two"):
            StratifiedRandomization().assign(data, _spec(), np.random.default_rng(7))

    def test_stratified_randomization_rejects_infeasible_share(self) -> None:
        data = _design_data(20)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            StratifiedRandomization(treatment_share=0.05).assign(
                data, _spec(), np.random.default_rng(7)
            )

    def test_augment_does_not_mutate_source(self) -> None:
        data = _design_data(20)
        result = CompleteRandomization().assign(data, _spec(), np.random.default_rng(3))
        augmented = result.augment(data, "assigned")
        self.assertNotIn("assigned", data)
        self.assertIn("assigned", augmented)


class RerandomizationTests(unittest.TestCase):
    def _assert_common_contract(self, mechanism: object) -> None:
        data = _design_data()
        first = mechanism.assign(data, _spec(), np.random.default_rng(91))
        second = mechanism.assign(data, _spec(), np.random.default_rng(91))
        self.assertEqual(first.n_treatment, 60)
        self.assertEqual(first.n_control, 60)
        self.assertLessEqual(first.attempts, mechanism.max_attempts)
        self.assertIn("acceptance_threshold", first.diagnostics)
        self.assertIn("fallback_used", first.diagnostics)
        self.assertIsNotNone(first.replay_trace)
        self.assertEqual(
            set(first.replay_trace.seeds),
            {"pilot_seed", "candidate_seed", "tie_seed", "fallback_seed"},
        )
        self.assertEqual(
            first.diagnostics["feature_cols"],
            ("x", "z", "count_with_ties", "category"),
        )
        pd.testing.assert_series_equal(first.labels, second.labels)
        self.assertEqual(first.diagnostics["accepted_score"], second.diagnostics["accepted_score"])

    def test_mahalanobis_contract(self) -> None:
        self._assert_common_contract(
            RerandomizationMahalanobis(n_pilot=40, max_attempts=80)
        )

    def test_max_smd_contract(self) -> None:
        self._assert_common_contract(RerandomizationMaxSMD(n_pilot=40, max_attempts=80))

    def test_mixed_balance_contract_with_ties_and_missingness(self) -> None:
        mechanism = RerandomizationMixedBalance(n_pilot=40, max_attempts=80)
        self._assert_common_contract(mechanism)
        result = mechanism.assign(_design_data(), _spec(), np.random.default_rng(12))
        self.assertIn("accepted_numeric_max_smd", result.diagnostics)
        self.assertIn("accepted_categorical_max_smd", result.diagnostics)
        self.assertIn("accepted_max_ks_distance", result.diagnostics)
        self.assertFalse(result.diagnostics["anti_nuclear"])

    def test_anti_nuclear_tie_spreading_is_deterministic(self) -> None:
        data = _design_data()
        mechanism = RerandomizationMixedBalance(
            feature_cols=("count_with_ties",),
            shape_cols=("count_with_ties",),
            anti_nuclear=True,
            n_pilot=40,
            max_attempts=80,
        )
        first = mechanism.assign(data, _spec(), np.random.default_rng(5))
        second = mechanism.assign(data, _spec(), np.random.default_rng(5))
        pd.testing.assert_series_equal(first.labels, second.labels)
        self.assertTrue(first.diagnostics["anti_nuclear"])

    def test_outcome_cannot_be_selected(self) -> None:
        data = _design_data()
        mechanism = RerandomizationMaxSMD(feature_cols=("y",), n_pilot=20)
        with self.assertRaisesRegex(ValueError, "pre-treatment"):
            mechanism.assign(data, _spec(), np.random.default_rng(1))

    def test_changing_outcome_does_not_change_assignment(self) -> None:
        data = _design_data()
        changed = data.copy()
        changed["y"] = np.arange(len(changed)) ** 3
        mechanism = RerandomizationMixedBalance(n_pilot=40, max_attempts=80)
        first = mechanism.assign(data, _spec(), np.random.default_rng(33))
        second = mechanism.assign(changed, _spec(), np.random.default_rng(33))
        pd.testing.assert_series_equal(first.labels, second.labels)

    def test_fallback_is_reported(self) -> None:
        data = _design_data(40)
        mechanism = RerandomizationMaxSMD(
            acceptance_probability=0.01,
            n_pilot=20,
            max_attempts=1,
            fallback_policy="best",
        )
        result = mechanism.assign(data, _spec(), np.random.default_rng(8))
        self.assertIn("fallback_used", result.diagnostics)
        if result.diagnostics["fallback_used"]:
            self.assertEqual(result.diagnostics["fallback_policy"], "best")
            self.assertTrue(result.mechanism.endswith("__best_of_k_fallback"))
            self.assertEqual(result.replay_trace.fallback_design, "best_of_k")

    def test_complete_randomization_fallback_is_a_distinct_design(self) -> None:
        class NeverAccept(RerandomizationMaxSMD):
            def _calibrate(self, pilot_rng, n_rows, prepared, n_treatment):
                return -1.0, {}, None

        result = NeverAccept(
            n_pilot=20,
            max_attempts=2,
            fallback_policy="complete_randomization",
        ).assign(_design_data(40), _spec(), np.random.default_rng(8))
        self.assertEqual(
            result.mechanism,
            "rerandomization_max_smd__complete_randomization_fallback",
        )
        self.assertEqual(result.replay_trace.fallback_design, "complete_randomization")
        self.assertIsNone(result.replay_trace.selected_candidate_attempt)


class HypExAssignmentTests(unittest.TestCase):
    def test_hypex_best_of_k_is_reproducible_and_traced(self) -> None:
        data = _design_data(80)
        mechanism = HypExAABestOfK(
            feature_cols=("x", "z"), n_iterations=3
        )
        first = mechanism.assign(data, _spec(), np.random.default_rng(19))
        second = mechanism.assign(data, _spec(), np.random.default_rng(19))
        pd.testing.assert_series_equal(first.labels, second.labels)
        self.assertEqual(first.mechanism, "hypex_aa_best_of_k")
        self.assertEqual(len(first.replay_trace.seeds["hypex_random_states"]), 3)
        self.assertEqual(first.replay_trace.attempts, 3)
        self.assertIsNotNone(first.replay_trace.selected_candidate_attempt)
        self.assertFalse(first.replay_trace.fallback_used)


if __name__ == "__main__":
    unittest.main()
