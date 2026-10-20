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


    def test_nonnull_h0_flagged(self) -> None:
        # population_shift-like H0: the effect parameter is 0 but the realised
        # SATE is 0.285, so "p <= alpha" is power, not size.
        rng = np.random.default_rng(23)
        n = 400
        truth = 0.285 + rng.normal(0.0, 0.005, size=n)
        effect = truth + rng.normal(size=n) * 0.1
        se = np.full(n, 0.1)
        p = 2.0 * __import__("scipy").stats.norm.sf(np.abs(effect / se))
        h0_frame = pd.DataFrame(
            {
                "effect": effect,
                "standard_error": se,
                "p_value": p,
                "ci_low": effect - 1.96 * se,
                "ci_high": effect + 1.96 * se,
                "true_sample_ate": truth,
            }
        )
        h1_frame = h0_frame.copy()
        for column in ("effect", "ci_low", "ci_high", "true_sample_ate"):
            h1_frame[column] = h1_frame[column] + 0.2
        h0 = summarize_trial_group(h0_frame, estimand_id="sample_ate")
        h1 = summarize_trial_group(h1_frame, estimand_id="sample_ate")
        self.assertFalse(h0["truth_is_null"])
        self.assertGreater(float(h0["rejection_rate"]), 0.5)  # test against 0 is power
        quality = evaluate_inference_quality(
            h0, h1, theory_grade=TheoryGrade.A, causal_safety=CausalSafety.SAFE
        )
        self.assertFalse(quality["h0_is_null"])
        self.assertEqual(quality["size_metric"], "rejection_vs_truth")
        self.assertLess(float(quality["size"]), 0.08)
        self.assertEqual(quality["pvalue_state"], "not_applicable")
        self.assertNotEqual(quality["inference_quality"], "bad")

    def test_exactly_null_h0_uses_rejection_against_zero(self) -> None:
        h0_frame, h1_frame = _trials(seed=29, fpr=0.05)
        h0 = summarize_trial_group(h0_frame, estimand_id="sample_ate")
        h1 = summarize_trial_group(h1_frame, estimand_id="sample_ate")
        self.assertTrue(h0["truth_is_null"])
        quality = evaluate_inference_quality(
            h0, h1, theory_grade=TheoryGrade.A, causal_safety=CausalSafety.SAFE
        )
        self.assertTrue(quality["h0_is_null"])
        self.assertEqual(quality["size_metric"], "rejection_vs_zero")
        self.assertEqual(float(quality["size"]), float(h0["rejection_rate"]))

    def test_point_bias_without_mc_evidence_is_uncertain_not_bad(self) -> None:
        h0 = {
            "truth_is_null": True,
            "rejection_rate": 0.05,
            "rejection_ci_low": 0.027,
            "rejection_ci_high": 0.09,
            "pvalue_superuniform_p": 0.5,
        }
        h1 = {
            "n_trials": 200,
            "empirical_sd": 1.0,
            "bias": 0.13,
            "bias_mc_se": 1.0 / np.sqrt(200),
            "coverage": 0.95,
            "coverage_ci_low": 0.91,
            "coverage_ci_high": 0.97,
            "se_to_sd": 1.0,
            "se_to_sd_ci_low": 0.9,
            "se_to_sd_ci_high": 1.1,
        }
        quality = evaluate_inference_quality(
            h0, h1, theory_grade=TheoryGrade.A, causal_safety=CausalSafety.SAFE
        )
        self.assertEqual(quality["bias_state"], "uncertain")
        self.assertEqual(quality["inference_quality"], "uncertain")
        h1["bias"] = 0.30
        quality = evaluate_inference_quality(
            h0, h1, theory_grade=TheoryGrade.A, causal_safety=CausalSafety.SAFE
        )
        self.assertEqual(quality["inference_quality"], "bad")
        self.assertIn("bias", quality["verdict_reasons"])


if __name__ == "__main__":
    unittest.main()
