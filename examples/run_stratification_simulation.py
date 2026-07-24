from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar import (
    ABProportionalVarianceObjective,
    ExperimentSpec,
    GeneticStrataReducer,
    run_stratification_race,
    summarize_stratification_trials,
)
from varwar.synthetic import (
    SyntheticConfig,
    generate_synthetic_ab,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    with (ROOT / "configs" / "stratification_simulation.toml").open("rb") as stream:
        raw = tomllib.load(stream)
    simulation = raw["simulation"]
    settings = raw["stratification"]
    n_trials = args.trials or int(simulation["n_trials"])
    category_cols = tuple(settings["category_cols"])
    spec = ExperimentSpec(
        pre_target_col="y_pre",
        pre_treatment_cols=("y_pre", "x1", "x2"),
        expert_safe_in_time_cols=("calendar_index", "market_index"),
        gated_in_time_cols=("operational_load",),
    )

    summaries: list[pd.DataFrame] = []
    for effect in simulation["effects"]:
        trials: list[pd.DataFrame] = []
        for trial in range(n_trials):
            seed = int(simulation["seed"]) + trial
            config = SyntheticConfig(
                n=int(simulation["n"]),
                effect=float(effect),
                seed=seed,
            )
            experiment = generate_synthetic_ab(config)
            common = dict(
                category_cols=category_cols,
                reference_target_col="y_pre",
                n_strata=int(settings["n_strata"]),
                population_size=int(settings["population_size"]),
                max_generations=int(settings["max_generations"]),
                elite_size=int(settings["elite_size"]),
                patience=int(settings["patience"]),
                unknown_policy="largest",
                output_col="__varwar_race_stratum",
                random_state=seed,
            )
            reducers = {
                "genetic_strata_reducer": GeneticStrataReducer(
                    objective=ABProportionalVarianceObjective(),
                    **common,
                ),
            }
            result = run_stratification_race(
                experiment,
                experiment,
                spec,
                category_cols=category_cols,
                reference_target_col="y_pre",
                n_strata=int(settings["n_strata"]),
                reducers=reducers,
                validation_data=None,
                true_effect=float(effect),
                random_state=seed,
            )
            result["trial"] = trial
            trials.append(result)
        summary = summarize_stratification_trials(
            pd.concat(trials, ignore_index=True),
            true_effect=float(effect),
        )
        summary.insert(0, "scenario", "null" if float(effect) == 0.0 else "alternative")
        summaries.append(summary)

    output = pd.concat(summaries, ignore_index=True)
    columns = [
        "scenario",
        "pipeline",
        "objective",
        "n_trials",
        "success_rate",
        "bias",
        "rmse",
        "mean_standard_error",
        "empirical_standard_deviation",
        "rejection_rate",
        "rate_kind",
    ]
    print(output.reindex(columns=columns).to_string(index=False))


if __name__ == "__main__":
    main()
