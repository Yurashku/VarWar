"""Monte Carlo behaviour of the verdict rule (docs/methods/statistical_validation.md).

The rule must not turn Monte Carlo noise into ``bad``: an unbiased, calibrated
estimator at R=200 is ``bad`` in well under 2% of synthetic aggregates, while
real defects (bias of 0.3 SD, FPR 0.12) are detected.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from varwar.analysis import CausalSafety, TheoryGrade
from varwar.validation import (
    benjamini_hochberg_reject,
    evaluate_inference_quality,
    summarize_trial_group,
)


def _frame(
    rng: np.random.Generator,
    n: int,
    *,
    truth: float,
    bias: float = 0.0,
    se_scale: float = 1.0,
) -> pd.DataFrame:
    effect = truth + bias + rng.normal(size=n)
    se = np.full(n, se_scale)
    return pd.DataFrame(
        {
            "effect": effect,
            "standard_error": se,
            "p_value": 2.0 * stats.norm.sf(np.abs(effect / se)),
            "ci_low": effect - 1.96 * se,
            "ci_high": effect + 1.96 * se,
            "true_sample_ate": np.full(n, truth),
        }
    )


def _stage_verdicts(
    seed: int,
    n_aggregates: int,
    n_trials: int,
    *,
    bias: float = 0.0,
) -> pd.Series:
    """Apply the full stage rule, including BH across rows, to synthetic rows."""

    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_aggregates):
        h0 = summarize_trial_group(_frame(rng, n_trials, truth=0.0, bias=bias), estimand_id="sample_ate")
        h1 = summarize_trial_group(_frame(rng, n_trials, truth=0.2, bias=bias), estimand_id="sample_ate")
        pairs.append((h0, h1))
    rejected = benjamini_hochberg_reject(
        [h0["pvalue_superuniform_p"] for h0, _ in pairs], q=0.05
    )
    verdicts = [
        evaluate_inference_quality(
            h0,
            h1,
            theory_grade=TheoryGrade.A,
            causal_safety=CausalSafety.SAFE,
            superuniformity_rejected=bool(flag),
        )["inference_quality"]
        for (h0, h1), flag in zip(pairs, rejected)
    ]
    return pd.Series(verdicts)


class MonteCarloVerdictTests(unittest.TestCase):
    def test_unbiased_calibrated_estimator_is_rarely_bad_at_r200(self) -> None:
        verdicts = _stage_verdicts(20261001, n_aggregates=600, n_trials=200)
        bad_share = float(verdicts.eq("bad").mean())
        self.assertLess(bad_share, 0.02)
        # The old point rule failed ~16% of such rows on bias alone; the
        # Monte Carlo-aware rule keeps most of them good.
        self.assertGreater(float(verdicts.eq("good").mean()), 0.5)

    def test_bias_of_030_sd_is_flagged(self) -> None:
        r200 = _stage_verdicts(20261002, n_aggregates=200, n_trials=200, bias=0.3)
        self.assertGreater(float(r200.isin(["bad", "uncertain"]).mean()), 0.95)
        # |b| - 1.96 MC SE > 0.10 SD needs |b|/SD > 0.239 at R=200, which a
        # 0.3 SD bias reaches with probability ~0.81; at R=500 it is ~0.99.
        self.assertGreater(float(r200.eq("bad").mean()), 0.70)
        r500 = _stage_verdicts(20261003, n_aggregates=100, n_trials=500, bias=0.3)
        self.assertGreater(float(r500.eq("bad").mean()), 0.90)

    def test_fpr_012_at_r1000_is_bad(self) -> None:
        rng = np.random.default_rng(20261004)
        h0_frame = _frame(rng, 1000, truth=0.0)
        rejected = rng.choice(1000, size=120, replace=False)
        h0_frame["p_value"] = np.where(
            np.isin(np.arange(1000), rejected), 0.001, rng.uniform(0.06, 1.0, size=1000)
        )
        h0 = summarize_trial_group(h0_frame, estimand_id="sample_ate")
        h1 = summarize_trial_group(_frame(rng, 1000, truth=0.2), estimand_id="sample_ate")
        quality = evaluate_inference_quality(
            h0, h1, theory_grade=TheoryGrade.A, causal_safety=CausalSafety.SAFE
        )
        self.assertAlmostEqual(float(h0["rejection_rate"]), 0.12)
        self.assertEqual(quality["inference_quality"], "bad")
        self.assertIn("fpr", str(quality["verdict_reasons"]).split(";"))

    def test_conservative_se_is_a_flag_not_a_failure(self) -> None:
        rng = np.random.default_rng(20261005)
        h0 = summarize_trial_group(_frame(rng, 1000, truth=0.0, se_scale=1.3), estimand_id="sample_ate")
        h1 = summarize_trial_group(_frame(rng, 1000, truth=0.2, se_scale=1.3), estimand_id="sample_ate")
        quality = evaluate_inference_quality(
            h0, h1, theory_grade=TheoryGrade.A, causal_safety=CausalSafety.SAFE
        )
        self.assertTrue(quality["conservative"])
        self.assertEqual(quality["se_sd_state"], "conservative")
        self.assertNotEqual(quality["inference_quality"], "bad")

    def test_understated_se_is_bad(self) -> None:
        rng = np.random.default_rng(20261006)
        h0 = summarize_trial_group(_frame(rng, 500, truth=0.0, se_scale=0.6), estimand_id="sample_ate")
        h1 = summarize_trial_group(_frame(rng, 500, truth=0.2, se_scale=0.6), estimand_id="sample_ate")
        quality = evaluate_inference_quality(
            h0, h1, theory_grade=TheoryGrade.A, causal_safety=CausalSafety.SAFE
        )
        self.assertEqual(quality["inference_quality"], "bad")
        self.assertIn("se_sd", str(quality["verdict_reasons"]))

    def test_acceptance_criteria_confirm_calibrated_estimator_at_r2000(self) -> None:
        rng = np.random.default_rng(20261007)
        confirmed = []
        for bias in (0.0,) * 20 + (0.3,) * 5:
            h0 = summarize_trial_group(_frame(rng, 2000, truth=0.0, bias=bias), estimand_id="sample_ate")
            h1 = summarize_trial_group(_frame(rng, 2000, truth=0.2, bias=bias), estimand_id="sample_ate")
            quality = evaluate_inference_quality(
                h0, h1, theory_grade=TheoryGrade.A, causal_safety=CausalSafety.SAFE
            )
            confirmed.append(bool(quality["validity_confirmed"]))
        self.assertGreaterEqual(sum(confirmed[:20]), 15)
        self.assertFalse(any(confirmed[20:]))

    def test_benjamini_hochberg_matches_definition(self) -> None:
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
        reject = benjamini_hochberg_reject(p, q=0.05)
        # k = max{i : p_(i) <= i q / m} with m = 8 finite p-values -> i = 2.
        self.assertEqual(reject.tolist(), [True, True] + [False] * 7)


if __name__ == "__main__":
    unittest.main()
