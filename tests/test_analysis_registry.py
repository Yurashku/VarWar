from __future__ import annotations

import unittest

from varwar.analysis import PipelineStatus, RaceContext
from varwar.analysis_registry import build_analysis_method_registry


class AnalysisRegistryTests(unittest.TestCase):
    def test_required_analysis_families_are_registered(self) -> None:
        methods = build_analysis_method_registry(
            resamples=99,
            prefer_pypi_hypex=False,
            population_weights={"low": 0.5, "high": 0.5},
            population_moments={"score": 0.0},
        )
        required = {
            "diff_means__welch",
            "diff_means__student_pooled",
            "diff_means__design_based_normal",
            "diff_means__welch_studentized_randomization",
            "diff_means__bootstrap_t",
            "diff_means__sharp_null_randomization",
            "lin_regression__hc2",
            "ml_rate__crossfit_hc2",
            "trimmed_mean__yuen",
            "winsorized_mean__full_refit_bootstrap",
            "median_of_means_ate__bootstrap",
            "catoni_mean_ate__bootstrap",
            "nearest_neighbor_matching_ate__reuse_aware",
            "optimal_transport_matching_ate__point",
            "population_weighted_stratified_ate__native",
        }
        self.assertTrue(required.issubset(methods))

    def test_ot_is_visible_as_point_only(self) -> None:
        method = build_analysis_method_registry(
            resamples=99,
            prefer_pypi_hypex=False,
        )["optimal_transport_matching_ate__point"]
        status, reason = method.compatibility(RaceContext())
        self.assertEqual(status, PipelineStatus.POINT_ONLY)
        self.assertIn("inference", reason)


if __name__ == "__main__":
    unittest.main()
