from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar import ExperimentSpec, NumericFeatureBinner, run_stratification_race
from varwar.synthetic import SyntheticConfig, generate_synthetic_ab


def main() -> None:
    config_path = ROOT / "configs" / "stratification_race.toml"
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    synthetic = raw["synthetic"]
    base_config = SyntheticConfig(
        n=int(synthetic["n"]),
        effect=float(synthetic["effect"]),
        seed=int(synthetic["seed"]),
    )
    experiment = generate_synthetic_ab(base_config)

    binning = raw["binning"]
    binner = NumericFeatureBinner(
        columns=tuple(binning["columns"]),
        strategy=str(binning["strategy"]),
        n_bins=int(binning["n_bins"]),
    ).fit(experiment)
    experiment = binner.augment(experiment)

    race = raw["stratification"]
    spec = ExperimentSpec(
        pre_target_col="y_pre",
        pre_treatment_cols=("y_pre", "x1", "x2"),
        expert_safe_in_time_cols=("calendar_index", "market_index"),
        gated_in_time_cols=("operational_load",),
    )
    results = run_stratification_race(
        experiment,
        experiment,
        spec,
        category_cols=tuple(race["category_cols"]),
        reference_target_col="y_pre",
        n_strata=int(race["n_strata"]),
        measure_memory=bool(race["measure_memory"]),
        random_state=int(race["random_state"]),
        validation_data=None,
        true_effect=base_config.effect,
    )
    columns = [
        "pipeline",
        "objective",
        "status",
        "effect",
        "standard_error",
        "p_value",
        "bias",
        "preprocess_runtime_ms",
        "estimator_runtime_ms",
        "preprocess_peak_memory_bytes",
        "train_score",
        "validation_score",
        "n_atomic_strata",
        "n_strata",
    ]
    print(results.reindex(columns=columns).to_string(index=False))


if __name__ == "__main__":
    main()
