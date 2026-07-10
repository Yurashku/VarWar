from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar.benchmark import run_estimator_race
from varwar.core import ExperimentSpec
from varwar.synthetic import SyntheticConfig, generate_historical_controls, generate_synthetic_ab


def main() -> None:
    config = SyntheticConfig(n=5000, effect=0.25, seed=42)
    data = generate_synthetic_ab(config)
    historical = generate_historical_controls(config)
    spec = ExperimentSpec(
        covariate_cols=("y_pre", "x1", "x2"),
        pre_treatment_cols=("y_pre", "x1", "x2"),
        expert_safe_in_time_cols=("calendar_index", "market_index"),
        gated_in_time_cols=("operational_load",),
        unsafe_demo_cols=("post_outcome_proxy",),
        pre_target_col="y_pre",
        numerator_col="numerator",
        denominator_col="denominator",
        stratum_col="stratum",
        instrument_col="assigned",
        received_treatment_col="received",
    )
    results = run_estimator_race(data, spec, auxiliary_data=historical)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
