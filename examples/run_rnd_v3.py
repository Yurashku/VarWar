"""Запуск стадии VarWar R&D v3 (целиком или одним шардом по мирам/пайплайнам).

Seed каждой реализации зависит только от (--seed, world_id, trial), поэтому
шард по любому подмножеству миров даёт те же числа, что и общий прогон.
Шарды склеиваются ``examples/merge_rnd_shards.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar.rnd_runner import RNDRunConfig, run_rnd
from varwar.synthetic_v3 import WORLD_CATALOG


#: Stage defaults of the v3.1 plan (docs/review/v3_review_design.md, §D.1):
#: (trials, worlds, effects, max matrix rows, seed).
STAGE_DEFAULTS = {
    "smoke": (5, ("gaussian_r2_04_n200", "contaminated"), (0.0, 0.2), 48, 20261001),
    "screening": (500, tuple(WORLD_CATALOG), (0.0, 0.1, 0.2), None, 20261010),
    "confirmation": (2000, tuple(WORLD_CATALOG), (0.0, 0.1, 0.2), None, 20261020),
    "power": (1000, tuple(WORLD_CATALOG), (0.0, 0.05, 0.1, 0.15, 0.2, 0.3), None, 20261030),
}

EPILOG = """\
Шардирование по мирам (seed не зависит от состава шарда):

  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python examples/run_rnd_v3.py --list-worlds | tr ' ' '\\n' | \\
    xargs -P 6 -I{} python examples/run_rnd_v3.py --stage screening \\
      --world {} --pipelines-file configs/rnd_v3_1_screening_pipelines.txt \\
      --output-dir results/rnd_v3_1/shards/{} --resume
  # -P ограничивает число параллельных шардов по RAM (~1.5 ГБ на процесс).
  python examples/merge_rnd_shards.py results/rnd_v3_1/shards/* \\
      --stage screening --output results/rnd_v3_1/screening

Финалисты confirmation берутся из метаданных screening:

  python examples/run_rnd_v3.py --stage confirmation --world W \\
      --pipelines-file results/rnd_v3_1/screening/run_metadata.json ...

Результат пишется в <output-dir>/<stage>/. С --resume стадия пропускается,
если там уже есть завершённый прогон с той же конфигурацией (иначе ошибка).
"""


def _parse_effects(values: list[str] | None) -> tuple[float, ...] | None:
    if not values:
        return None
    effects: list[float] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                effects.append(float(part))
    return tuple(dict.fromkeys(effects))


def _read_pipelines_file(path: Path) -> tuple[str, ...]:
    """One ID per line (``#`` comments), a JSON list, or run_metadata.json."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            if "finalists" not in payload:
                raise SystemExit(f"{path}: JSON object has no 'finalists' list.")
            payload = payload["finalists"]
        if not isinstance(payload, list):
            raise SystemExit(f"{path}: expected a JSON list of pipeline IDs.")
        return tuple(str(item) for item in payload)
    ids = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return tuple(ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VarWar scientific R&D v3: одна стадия или её шард.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_DEFAULTS),
        default="smoke",
        help="Стадия плана; задаёт значения по умолчанию R, seed, миров и эффектов.",
    )
    parser.add_argument("--trials", type=int, help="Число реализаций Монте-Карло R.")
    parser.add_argument("--seed", type=int, help="Базовый seed стадии.")
    parser.add_argument(
        "--world",
        action="append",
        dest="worlds",
        help="ID мира; можно повторять. По умолчанию — миры стадии.",
    )
    parser.add_argument(
        "--effects",
        action="append",
        help="Сетка эффектов в σ через запятую (должна содержать 0), например 0,0.1,0.2.",
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        dest="pipelines",
        help="Канонический ID пайплайна; параметр можно повторять.",
    )
    parser.add_argument(
        "--pipelines-file",
        type=Path,
        help="Файл с ID пайплайнов: по одному в строке (# — комментарий), "
        "JSON-список или run_metadata.json стадии (берётся 'finalists').",
    )
    parser.add_argument(
        "--max-pipelines",
        type=int,
        help="Ограничить число исполняемых строк матрицы (greedy покрытие компонентов).",
    )
    parser.add_argument(
        "--include-hypex-assignment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Включить дорогой интеграционный HypEx A/A best-of-K механизм.",
    )
    parser.add_argument(
        "--broadcast-mode-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Исполнять пайплайн, не читающий panel prior и in-time колонки, "
        "один раз и копировать результат в остальные контексты (по умолчанию да).",
    )
    parser.add_argument(
        "--memory-trials",
        type=int,
        default=2,
        help="Сколько первых реализаций каждого (мир, эффект) мерить tracemalloc.",
    )
    parser.add_argument(
        "--effect-modifier-centering",
        choices=("population", "cohort"),
        help="Принудительное центрирование модификатора эффекта для всех estimand. По умолчанию "
        "sample_ate-строки используют мир с центрированием по когорте, population_ate — по "
        "популяции (точная H0 в обоих треках).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "rnd_v3",
        help="Базовый каталог; результаты пишутся в <output-dir>/<stage>/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Пропустить стадию, если в каталоге уже есть завершённый прогон с той же конфигурацией.",
    )
    parser.add_argument(
        "--list-worlds",
        action="store_true",
        help="Напечатать ID миров каталога через пробел и выйти.",
    )
    return parser


def build_config(args: argparse.Namespace) -> RNDRunConfig:
    trials, worlds, effects, limit, seed = STAGE_DEFAULTS[args.stage]
    pipelines = list(args.pipelines or ())
    if args.pipelines_file is not None:
        pipelines.extend(_read_pipelines_file(args.pipelines_file))
    unknown = [world for world in (args.worlds or ()) if world not in WORLD_CATALOG]
    if unknown:
        raise SystemExit(f"Unknown world(s): {unknown}. See --list-worlds.")
    explicit_pipelines = tuple(dict.fromkeys(pipelines))
    return RNDRunConfig(
        stage=args.stage,
        n_trials=args.trials or trials,
        seed=args.seed if args.seed is not None else seed,
        world_ids=tuple(dict.fromkeys(args.worlds or worlds)),
        effects=_parse_effects(args.effects) or effects,
        max_executable_pipelines=(
            args.max_pipelines
            if args.max_pipelines is not None
            else (None if explicit_pipelines else limit)
        ),
        pipeline_ids=explicit_pipelines,
        include_hypex_assignment=(
            args.stage != "smoke"
            if args.include_hypex_assignment is None
            else args.include_hypex_assignment
        ),
        fast_only=args.stage == "smoke",
        broadcast_mode_invariant=args.broadcast_mode_invariant,
        memory_profile_trials=args.memory_trials,
        effect_modifier_centering=args.effect_modifier_centering,
    )


def completed_run_matches(output: Path, config: RNDRunConfig) -> bool:
    """True if ``output`` holds a finished run of exactly this config."""

    metadata_path = output / "run_metadata.json"
    if not (metadata_path.exists() and (output / "trial_results.parquet").exists()):
        return False
    stored = json.loads(metadata_path.read_text(encoding="utf-8")).get("config", {})
    current = json.loads(json.dumps(config.to_record()))
    if stored != current:
        differing = sorted(
            key for key in set(stored) | set(current) if stored.get(key) != current.get(key)
        )
        raise SystemExit(
            f"{output} already holds a run with a different config ({differing}); "
            "choose another --output-dir or delete it."
        )
    return True


def main() -> None:
    args = build_parser().parse_args()
    if args.list_worlds:
        print(" ".join(WORLD_CATALOG))
        return
    config = build_config(args)
    output = args.output_dir / args.stage
    if args.resume and completed_run_matches(output, config):
        print(f"Пропуск: {output} уже содержит завершённый прогон этой конфигурации.")
        return
    started = time.perf_counter()
    artifacts = run_rnd(config, output)
    print(f"Матрица: {len(artifacts.pipeline_matrix):,} строк")
    print(f"Прогоны: {len(artifacts.trial_results):,} строк")
    print(f"Агрегаты: {len(artifacts.aggregates):,} строк")
    print(f"Время: {time.perf_counter() - started:.1f} с")
    print(f"Результаты: {output}")


if __name__ == "__main__":
    main()
