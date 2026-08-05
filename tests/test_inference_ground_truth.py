from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from varwar.analysis import InformationMode, RaceContext, TemporalAccess
from varwar.analysis_registry import build_analysis_method_registry
from varwar.core import ExperimentSpec
from varwar.validation import summarize_trial_group


class InferenceGroundTruthTests(unittest.TestCase):
    def test_welch_has_numerically_calibrated_fpr_and_coverage(self) -> None:
        rng = np.random.default_rng(20260805)
        method = build_analysis_method_registry(
            resamples=99, prefer_pypi_hypex=False
        )["diff_means__welch"]
        spec = ExperimentSpec(pre_target_col=None)
        frames = {}
        for label, truth in (("H0", 0.0), ("H1", 0.2)):
            rows = []
            for _ in range(500):
                control = rng.normal(0.0, 1.0, 90)
                treated = rng.normal(truth, 1.6, 110)
                data = pd.DataFrame(
                    {
                        "treatment": np.r_[np.zeros(90), np.ones(110)],
                        "y": np.r_[control, treated],
                    }
                )
                result = method.analyze(
                    data,
                    spec,
                    RaceContext(
                        information_mode=InformationMode.EXPERIMENT_ONLY,
                        temporal_access=TemporalAccess.PRE_TREATMENT_ONLY,
                        ranking_estimand_id="sample_ate",
                    ),
                )
                rows.append(
                    {
                        **result.to_record(),
                        "true_sample_ate": truth,
                        "true_population_ate": truth,
                        "true_native_estimand": np.nan,
                    }
                )
            frames[label] = summarize_trial_group(
                pd.DataFrame(rows), estimand_id="sample_ate"
            )
        h0 = frames["H0"]
        summary = frames["H1"]
        self.assertGreater(float(h0["rejection_rate"]), 0.02)
        self.assertLess(float(h0["rejection_rate"]), 0.08)
        self.assertTrue(bool(h0["pvalue_superuniform_pass"]))
        self.assertLess(abs(float(summary["bias"])), 0.03)
        self.assertGreater(float(summary["coverage"]), 0.92)
        self.assertLess(float(summary["coverage"]), 0.98)
        self.assertGreater(float(summary["se_to_sd"]), 0.88)
        self.assertLess(float(summary["se_to_sd"]), 1.12)


if __name__ == "__main__":
    unittest.main()
