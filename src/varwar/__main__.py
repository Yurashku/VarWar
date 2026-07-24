from __future__ import annotations

import argparse
from pathlib import Path

from varwar.benchmark import run_estimator_race
from varwar.core import ExperimentSpec
from varwar.synthetic import SyntheticConfig, generate_synthetic_ab


def run_synthetic(args: argparse.Namespace) -> None:
    config = SyntheticConfig(
        n=args.n,
        effect=args.effect,
        covariate_signal=args.covariate_signal,
        noise_scale=args.noise_scale,
        seed=args.seed,
        heavy_tail=args.heavy_tail,
    )
    data = generate_synthetic_ab(config)
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
    result = run_estimator_race(
        data,
        spec,
        auxiliary_data=None,
        measure_memory=args.measure_memory,
    )
    console_columns = [
        "estimator",
        "status",
        "effect",
        "standard_error",
        "p_value",
        "ci_low",
        "ci_high",
        "n_control",
        "n_treatment",
        "runtime_ms",
        "diag_backend",
    ]
    print(result.reindex(columns=console_columns).to_string(index=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="varwar")
    sub = parser.add_subparsers(dest="command", required=True)
    synth = sub.add_parser("run-synthetic")
    synth.add_argument("--n", type=int, default=5000)
    synth.add_argument("--effect", type=float, default=0.25)
    synth.add_argument("--covariate-signal", type=float, default=0.75)
    synth.add_argument("--noise-scale", type=float, default=1.0)
    synth.add_argument("--seed", type=int, default=42)
    synth.add_argument("--heavy-tail", action="store_true")
    synth.add_argument("--measure-memory", action="store_true")
    synth.add_argument("--output", default="")
    synth.set_defaults(func=run_synthetic)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
