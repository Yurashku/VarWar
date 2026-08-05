from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from varwar.analysis import CausalSafety, TheoryGrade
from varwar.validation import evaluate_inference_quality, summarize_trial_group


def _trials(*, seed: int, fpr: float, n: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p = 2.0 * __import__("scipy").stats.norm.sf(np.abs(z))
    if fpr > 0.5:
        p[: int(fpr * n)] = 0.001
    effect0 = z
    h0 = pd.DataFrame(
        {
            "effect": effect0,
            "standard_error": np.ones(n),
            "p_value": p,
            "ci_low": effect0 - 1.96,
            "ci_high": effect0 + 1.96,
            "true_sample_ate": np.zeros(n),
            "true_population_ate": np.zeros(n),
            "true_native_estimand": np.zeros(n),
        }
    )
    effect1 = 0.2 + rng.normal(size=n)
    h1 = h0.copy()
    h1["effect"] = effect1
    h1["ci_low"] = effect1 - 1.96
    h1["ci_high"] = effect1 + 1.96
    h1["true_sample_ate"] = 0.2
    return h0, h1


class ValidationTests(unittest.TestCase):
    def test_ground_truth_metrics_are_numeric(self) -> None:
        h0_frame, h1_frame = _trials(seed=7, fpr=0.05)
        h0 = summarize_trial_group(h0_frame, estimand_id="sample_ate")
        h1 = summarize_trial_group(h1_frame, estimand_id="sample_ate")
        quality = evaluate_inference_quality(
            h0,
            h1,
            theory_grade=TheoryGrade.A,
            causal_safety=CausalSafety.SAFE,
        )
        for key in (
            "bias",
            "bias_ci_low",
            "bias_ci_high",
            "rmse",
            "se_to_sd",
            "coverage",
            "rejection_rate",
        ):
            self.assertTrue(np.isfinite(float((h1 if key != "rejection_rate" else h0)[key])))
        self.assertGreaterEqual(float(quality["inference_quality_score"]), 0.0)

    def test_catastrophic_fpr_is_bad(self) -> None:
        h0_frame, h1_frame = _trials(seed=11, fpr=0.9)
        h0 = summarize_trial_group(h0_frame, estimand_id="sample_ate")
        h1 = summarize_trial_group(h1_frame, estimand_id="sample_ate")
        quality = evaluate_inference_quality(
            h0,
            h1,
            theory_grade=TheoryGrade.A,
            causal_safety=CausalSafety.SAFE,
        )
        self.assertGreater(float(h0["rejection_rate"]), 0.8)
        self.assertEqual(quality["inference_quality"], "bad")
        self.assertFalse(quality["fpr_pass"])

    def test_native_estimand_and_ate_displacement_are_separate(self) -> None:
        h0_frame, h1_frame = _trials(seed=17, fpr=0.05)
        h1_frame["true_native_estimand"] = 0.0
        h1_frame["true_sample_ate"] = 0.2
        h1_frame["effect"] -= 0.2
        summary = summarize_trial_group(h1_frame, estimand_id="robust_native")
        self.assertAlmostEqual(float(summary["ate_displacement"]), -0.2, delta=0.1)
        self.assertLess(abs(float(summary["bias"])), 0.1)

    def test_zero_fpr_with_few_trials_is_uncertain_not_automatically_good(self) -> None:
        h0 = {
            "rejection_rate": 0.0,
            "rejection_ci_low": 0.0,
            "rejection_ci_high": 0.071,
            "pvalue_superuniform_pass": True,
        }
        h1 = {
            "empirical_sd": 1.0,
            "bias": 0.0,
            "coverage": 0.95,
            "coverage_ci_low": 0.91,
            "coverage_ci_high": 0.98,
            "se_to_sd": 1.0,
        }
        quality = evaluate_inference_quality(
            h0,
            h1,
            theory_grade=TheoryGrade.A,
            causal_safety=CausalSafety.SAFE,
        )
        self.assertTrue(quality["fpr_uncertain"])
        self.assertEqual(quality["inference_quality"], "warning")


if __name__ == "__main__":
    unittest.main()
