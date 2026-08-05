from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec
from varwar.estimators import (
    NearestNeighborMatchingEstimator,
    OptimalTransportMatchingEstimator,
)


def _asymmetric_matching_fixture() -> pd.DataFrame:
    # Forward effects are [2, 3, 1] (ATT=2); reverse effects in the original
    # treatment-minus-control orientation are [2, 1] (ATC=1.5).  Therefore
    # the sample-weighted symmetric ATE is 3/5*2 + 2/5*1.5 = 1.8.
    return pd.DataFrame(
        {
            "treatment": [0, 0, 1, 1, 1],
            "y": [0.0, 10.0, 2.0, 3.0, 11.0],
            "x": [0.0, 10.0, 0.0, 1.0, 9.0],
        }
    )


class MatchingEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = _asymmetric_matching_fixture()
        self.spec = ExperimentSpec(pre_treatment_cols=("x",))

    def test_nearest_neighbor_symmetric_ate_has_correct_reverse_sign(self) -> None:
        result = NearestNeighborMatchingEstimator(estimand="ate").estimate(
            self.data, self.spec
        )

        self.assertAlmostEqual(result.effect, 1.8)
        self.assertAlmostEqual(result.diagnostics["att_effect"], 2.0)
        self.assertAlmostEqual(result.diagnostics["atc_effect"], 1.5)
        self.assertEqual(
            result.diagnostics["inference"],
            "reuse_aware_linear_weight_sandwich",
        )
        self.assertEqual(
            result.diagnostics["att_atc_se_combination"],
            "not_used_joint_linear_weights",
        )
        self.assertEqual(result.diagnostics["inference_quality"], "approximate")

    def test_nearest_neighbor_ate_variance_uses_reuse_weights(self) -> None:
        result = NearestNeighborMatchingEstimator(estimand="ate").estimate(
            self.data, self.spec
        )

        y_treatment = np.array([2.0, 3.0, 11.0])
        y_control = np.array([0.0, 10.0])
        treatment_weights = np.array([2.0, 1.0, 2.0]) / 5.0
        control_weights = np.array([3.0, 2.0]) / 5.0

        def weighted_mean_variance(values: np.ndarray, weights: np.ndarray) -> float:
            mean = float(np.dot(weights, values))
            return float(
                np.dot(weights**2, (values - mean) ** 2)
                / (1.0 - np.dot(weights, weights))
            )

        expected = math.sqrt(
            weighted_mean_variance(y_treatment, treatment_weights)
            + weighted_mean_variance(y_control, control_weights)
        )
        self.assertAlmostEqual(result.standard_error, expected)
        self.assertEqual(result.diagnostics["max_control_reuse"], 2)
        self.assertEqual(result.diagnostics["max_treatment_reuse"], 1)

    def test_optimal_transport_symmetric_ate_has_correct_sign_and_is_point_only(self) -> None:
        result = OptimalTransportMatchingEstimator(estimand="ate").estimate(
            self.data, self.spec
        )

        self.assertAlmostEqual(result.effect, 1.8)
        self.assertAlmostEqual(result.diagnostics["att_effect"], 2.0)
        self.assertAlmostEqual(result.diagnostics["atc_effect"], 1.5)
        self.assertEqual(result.diagnostics["inference"], "point_only")
        self.assertFalse(result.diagnostics["transport_plan_uncertainty_accounted"])
        self.assertTrue(math.isnan(result.standard_error))
        self.assertTrue(math.isnan(result.p_value))
        self.assertTrue(math.isnan(result.ci_low))
        self.assertTrue(math.isnan(result.ci_high))

    def test_att_point_estimates_remain_in_treatment_minus_control_orientation(self) -> None:
        nearest = NearestNeighborMatchingEstimator(estimand="att").estimate(
            self.data, self.spec
        )
        transport = OptimalTransportMatchingEstimator(estimand="att").estimate(
            self.data, self.spec
        )

        self.assertAlmostEqual(nearest.effect, 2.0)
        self.assertAlmostEqual(transport.effect, 2.0)
        self.assertTrue(math.isfinite(nearest.standard_error))
        self.assertTrue(math.isfinite(transport.standard_error))


if __name__ == "__main__":
    unittest.main()
