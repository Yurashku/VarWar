"""VarWar estimator R&D toolkit."""

from varwar.benchmark import build_default_race, run_estimator_race
from varwar.core import ExperimentSpec, EstimateResult, WrappedEstimator
from varwar.estimators import (
    DifferenceInMeansEstimator,
    HypExABTestEstimator,
    HypExCupacEstimator,
    KSTestEstimator,
    PlaceholderEstimator,
    PermutationMeanTestEstimator,
    RandomizedTTestEstimator,
    RatioDeltaMethodEstimator,
    RegressionAdjustmentEstimator,
    StratifiedDifferenceInMeansEstimator,
    TrimmedMeanEstimator,
    VarianceWeightedStratifiedEstimator,
    WaldEstimator,
    WinsorizedMeanEstimator,
)
from varwar.transforms import CupedTransformer, LinearCupacTransformer, WinsorizeTransformer

__all__ = [
    "ExperimentSpec",
    "EstimateResult",
    "WrappedEstimator",
    "DifferenceInMeansEstimator",
    "HypExABTestEstimator",
    "HypExCupacEstimator",
    "KSTestEstimator",
    "PlaceholderEstimator",
    "PermutationMeanTestEstimator",
    "RandomizedTTestEstimator",
    "RatioDeltaMethodEstimator",
    "RegressionAdjustmentEstimator",
    "StratifiedDifferenceInMeansEstimator",
    "TrimmedMeanEstimator",
    "VarianceWeightedStratifiedEstimator",
    "WaldEstimator",
    "WinsorizedMeanEstimator",
    "CupedTransformer",
    "LinearCupacTransformer",
    "WinsorizeTransformer",
    "build_default_race",
    "run_estimator_race",
]
