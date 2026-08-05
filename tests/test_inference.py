from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from varwar.analysis import DirectPointEvaluator, RaceContext
from varwar.core import ExperimentSpec
from varwar.inference import (
    BootstrapTInference,
    DifferenceMeansPointEstimator,
    PooledStudentInference,
    StudentizedRandomizationInference,
    WelchInference,
)


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = np.array([0.0, 1.0, 2.0, 6.0])
        self.treated = np.array([1.0, 2.0, 4.0, 9.0, 12.0])
        self.data = pd.DataFrame(
            {
                "treatment": np.r_[
                    np.zeros(self.control.size),
                    np.ones(self.treated.size),
                ],
                "y": np.r_[self.control, self.treated],
            }
        )
        self.spec = ExperimentSpec()
        self.context = RaceContext()
        self.point = DifferenceMeansPointEstimator()
        self.evaluator = DirectPointEvaluator(self.point)
        self.observed = self.evaluator.evaluate_point(
            self.data,
            self.spec,
            self.context,
        )

    def test_welch_matches_scipy(self) -> None:
        result = WelchInference().infer(
            self.evaluator, self.observed, self.data, self.spec, self.context
        )
        reference = stats.ttest_ind(self.treated, self.control, equal_var=False)
        self.assertAlmostEqual(result.statistic, float(reference.statistic))
        self.assertAlmostEqual(result.p_value, float(reference.pvalue))

    def test_pooled_has_same_point_but_different_inference(self) -> None:
        welch = WelchInference().infer(
            self.evaluator, self.observed, self.data, self.spec, self.context
        )
        pooled = PooledStudentInference().infer(
            self.evaluator, self.observed, self.data, self.spec, self.context
        )
        self.assertNotAlmostEqual(welch.standard_error, pooled.standard_error)
        self.assertAlmostEqual(self.observed.effect, self.treated.mean() - self.control.mean())

    def test_resampling_is_reproducible(self) -> None:
        permutation = StudentizedRandomizationInference(n_resamples=99, random_state=7)
        first = permutation.infer(
            self.evaluator, self.observed, self.data, self.spec, self.context
        )
        second = permutation.infer(
            self.evaluator, self.observed, self.data, self.spec, self.context
        )
        self.assertEqual(first, second)
        bootstrap = BootstrapTInference(n_resamples=99, random_state=9)
        self.assertEqual(
            bootstrap.infer(self.evaluator, self.observed, self.data, self.spec, self.context),
            bootstrap.infer(self.evaluator, self.observed, self.data, self.spec, self.context),
        )


if __name__ == "__main__":
    unittest.main()
