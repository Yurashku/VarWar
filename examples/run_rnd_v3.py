from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar.rnd_runner import RNDRunConfig, run_rnd
from varwar.synthetic_v3 import WORLD_CATALOG


def main() -> None:
    parser = argparse.ArgumentParser(description="VarWar scientific R&D v3")
    parser.add_argument("--stage", choices=("smoke", "screening", "confirmation", "power"), default="smoke")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--world", action="append", dest="worlds")
    parser.add_argument(
        "--pipeline",
        action="append",
        dest="pipelines",
        help="Канонический ID пайплайна; параметр можно повторять.",
    )
    parser.add_argument("--max-pipelines", type=int)
    parser.add_argument(
        "--include-hypex-assignment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Включить дорогой интеграционный HypEx A/A best-of-K механизм.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rnd_v3")
    args = parser.parse_args()

    defaults = {
        "smoke": (5, ("gaussian_r2_04_n200", "contaminated"), (0.0, 0.2), 48, 20260805),
        "screening": (200, tuple(WORLD_CATALOG), (0.0, 0.1, 0.2), None, 20260806),
        "confirmation": (1000, tuple(WORLD_CATALOG), (0.0, 0.1, 0.2), None, 20260807),
        "power": (1000, tuple(WORLD_CATALOG), (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4), None, 20260808),
    }
    trials, worlds, effects, limit, seed = defaults[args.stage]
    config = RNDRunConfig(
        stage=args.stage,
        n_trials=args.trials or trials,
        seed=args.seed if args.seed is not None else seed,
        world_ids=tuple(args.worlds or worlds),
        effects=effects,
        max_executable_pipelines=(
            args.max_pipelines if args.max_pipelines is not None else limit
        ),
        pipeline_ids=tuple(args.pipelines or ()),
        include_hypex_assignment=(
            args.stage != "smoke"
            if args.include_hypex_assignment is None
            else args.include_hypex_assignment
        ),
        fast_only=args.stage == "smoke",
    )
    output = args.output_dir / args.stage
    artifacts = run_rnd(config, output)
    print(f"Матрица: {len(artifacts.pipeline_matrix):,} строк")
    print(f"Прогоны: {len(artifacts.trial_results):,} строк")
    print(f"Агрегаты: {len(artifacts.aggregates):,} строк")
    print(f"Результаты: {output}")


if __name__ == "__main__":
    main()
