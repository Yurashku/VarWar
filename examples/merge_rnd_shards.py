"""Склеить шарды одной стадии VarWar R&D v3 с проверкой целостности.

Проверяется, что
- у всех шардов совпадает конфигурация стадии (stage, seed, R, effects,
  resamples, ...) и версия кода (code_version, git SHA, хэш незакоммиченных
  правок в ``src``);
- ключ (world_id, trial, pipeline, information_mode, temporal_access,
  ranking_estimand_id, effect_size) уникален во всех шардах вместе;
- seed каждой реализации совпадает со схемой
  ``SeedSequence([seed, crc32(world_id), trial])``.

Агрегаты пересчитываются по объединённым trial-level данным, поэтому
BH-поправка super-uniformity действует на всю стадию.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar.rnd_report import write_rnd_report_ru
from varwar.rnd_runner import (
    SEED_COLUMNS,
    SEED_SCHEME,
    SHARD_INVARIANT_CONFIG_KEYS,
    aggregate_rnd_results,
    finalists_metadata,
    select_confirmation_finalists,
    select_rnd_leaders,
    trial_seeds,
    write_stage_payload,
)

UNIQUE_TRIAL_KEY = (
    "world_id",
    "trial",
    "pipeline",
    "information_mode",
    "temporal_access",
    "ranking_estimand_id",
    "effect_size",
)
PROVENANCE_KEYS = ("code_version", "git_sha", "git_src_diff_sha1")


class ShardMergeError(RuntimeError):
    """Shards are incomplete or inconsistent and must not be merged."""


def resolve_shard_dirs(paths: Sequence[Path], stage: str) -> list[Path]:
    """Accept stage dirs, ``<dir>/<stage>`` dirs or legacy ``<root>/shard_*/<stage>``."""

    resolved: list[Path] = []
    for path in paths:
        if (path / "run_metadata.json").exists() or (path / "trial_results.parquet").exists():
            resolved.append(path)
        elif (path / stage).is_dir():
            resolved.append(path / stage)
        else:
            legacy = sorted(path.glob(f"shard_*/{stage}"))
            if not legacy:
                raise ShardMergeError(f"{path}: no shard of stage {stage!r} found.")
            resolved.extend(legacy)
    unique = list(dict.fromkeys(item.resolve() for item in resolved))
    if not unique:
        raise ShardMergeError("No shard directories given.")
    return unique


def _load_metadata(shard: Path) -> dict[str, Any]:
    path = shard / "run_metadata.json"
    if not (shard / "trial_results.parquet").exists() or not path.exists():
        raise ShardMergeError(f"Incomplete shard (no trial_results/run_metadata): {shard}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if "config" not in metadata:
        raise ShardMergeError(f"{shard}: run_metadata.json has no 'config'.")
    return metadata


def validate_shard_metadata(
    metadata: Sequence[dict[str, Any]],
    shards: Sequence[Path],
    *,
    stage: str | None = None,
    allow_code_mismatch: bool = False,
) -> dict[str, Any]:
    """Return the common config or raise ``ShardMergeError``."""

    reference = metadata[0]["config"]
    problems: list[str] = []
    if stage is not None and reference.get("stage") != stage:
        problems.append(f"{shards[0]}: stage {reference.get('stage')!r} != {stage!r}")
    for shard, item in zip(shards[1:], metadata[1:]):
        config = item["config"]
        for key in SHARD_INVARIANT_CONFIG_KEYS:
            if config.get(key) != reference.get(key):
                problems.append(
                    f"{shard}: config[{key!r}]={config.get(key)!r} != {reference.get(key)!r} ({shards[0]})"
                )
        if not allow_code_mismatch:
            first = metadata[0].get("provenance", {})
            current = item.get("provenance", {})
            for key in PROVENANCE_KEYS:
                if current.get(key) != first.get(key):
                    problems.append(
                        f"{shard}: provenance[{key!r}]={current.get(key)!r} != {first.get(key)!r}"
                    )
    matrix_sizes = {item.get("n_matrix_rows") for item in metadata}
    if len(matrix_sizes) > 1:
        problems.append(f"pipeline matrix sizes differ between shards: {sorted(map(str, matrix_sizes))}")
    if problems:
        raise ShardMergeError("Shard configs disagree:\n  " + "\n  ".join(problems))
    return {key: reference.get(key) for key in SHARD_INVARIANT_CONFIG_KEYS}


def validate_trials(trials: pd.DataFrame, base_seed: int | None) -> None:
    missing = [column for column in UNIQUE_TRIAL_KEY if column not in trials]
    if missing:
        raise ShardMergeError(f"trial_results lack key columns: {missing}")
    duplicated = trials.duplicated(list(UNIQUE_TRIAL_KEY), keep=False)
    if duplicated.any():
        sample = trials.loc[duplicated, list(UNIQUE_TRIAL_KEY)].head(5).to_dict("records")
        raise ShardMergeError(
            f"{int(duplicated.sum())} trial rows share a key across shards, e.g. {sample}"
        )
    if base_seed is None or "seed" not in trials:
        return
    pairs = trials[["world_id", "trial", "seed"]].drop_duplicates()
    if pairs.duplicated(["world_id", "trial"]).any():
        raise ShardMergeError("A (world_id, trial) pair has more than one world seed.")
    expected = [
        trial_seeds(int(base_seed), str(world), int(trial))["world"]
        for world, trial in zip(pairs["world_id"], pairs["trial"])
    ]
    observed = pairs["seed"].astype("uint64").tolist()
    mismatched = [
        (world, trial)
        for world, trial, seen, want in zip(pairs["world_id"], pairs["trial"], observed, expected)
        if int(seen) != int(want)
    ]
    if mismatched:
        raise ShardMergeError(
            f"{len(mismatched)} (world, trial) seeds do not follow the seed scheme, e.g. {mismatched[:3]}"
        )


def merge_shards(
    shard_paths: Sequence[Path],
    output: Path,
    *,
    stage: str = "screening",
    allow_code_mismatch: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    shards = resolve_shard_dirs(shard_paths, stage)
    if output.resolve() in shards:
        raise ShardMergeError(f"Output {output} must differ from every shard directory.")
    metadata = [_load_metadata(shard) for shard in shards]
    common = validate_shard_metadata(
        metadata, shards, stage=stage, allow_code_mismatch=allow_code_mismatch
    )
    frames = []
    for shard in shards:
        frame = pd.read_parquet(shard / "trial_results.parquet")
        for column in SEED_COLUMNS:
            if column in frame:
                frame[column] = frame[column].astype("uint64")
        frame["shard"] = str(shard)
        frames.append(frame)
    trials = pd.concat(frames, ignore_index=True)
    validate_trials(trials, common.get("seed"))
    trials = trials.drop(columns="shard")

    output.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_rnd_results(trials)
    leaders = select_rnd_leaders(aggregates)
    finalists = select_confirmation_finalists(aggregates)
    trials.to_parquet(output / "trial_results.parquet", index=False)
    matrix_path = shards[0] / "pipeline_matrix.parquet"
    if matrix_path.exists():
        # The compatibility matrix is identical in every shard (checked by
        # size); copying avoids re-encoding 762k rows.
        shutil.copyfile(matrix_path, output / "pipeline_matrix.parquet")
        matrix = pd.read_parquet(matrix_path)
    else:
        matrix = pd.DataFrame(columns=["status"])
    aggregates.to_parquet(output / "aggregates.parquet", index=False)
    leaders.to_parquet(output / "leaders.parquet", index=False)
    aggregates.to_csv(output / "aggregates.csv", index=False, encoding="utf-8-sig")
    leaders.to_csv(output / "leaders.csv", index=False, encoding="utf-8-sig")
    write_stage_payload(output, aggregates, leaders, matrix)

    world_ids: list[str] = []
    pipeline_ids: list[str] = []
    for item in metadata:
        world_ids.extend(item["config"].get("world_ids", []))
        pipeline_ids.extend(item["config"].get("pipeline_ids", []))
    reference = metadata[0]["config"]
    merged_config = {
        **reference,
        **common,
        "world_ids": list(dict.fromkeys(world_ids)),
        "pipeline_ids": list(dict.fromkeys(pipeline_ids)),
    }
    merged = {
        "config": merged_config,
        "provenance": metadata[0].get("provenance", {}),
        "seed_scheme": SEED_SCHEME,
        "merged_from_shards": [
            {
                "path": str(shard),
                "world_ids": item["config"].get("world_ids", []),
                "pipeline_ids": item["config"].get("pipeline_ids", []),
                "n_trial_rows": item.get("n_trial_rows"),
                "provenance": item.get("provenance", {}),
            }
            for shard, item in zip(shards, metadata)
        ],
        "n_matrix_rows": metadata[0].get("n_matrix_rows"),
        "n_trial_rows": len(trials),
        "n_aggregate_rows": len(aggregates),
        "n_worlds": int(trials["world_id"].nunique()),
        "n_pipelines": int(trials["pipeline"].nunique()),
        "n_pipeline_executions": int(
            sum(int(item.get("n_pipeline_executions") or 0) for item in metadata)
        ),
        **finalists_metadata(finalists),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if write_report:
        write_rnd_report_ru(
            output / "report_ru.md", aggregates, trials, matrix, stage=stage
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Склеить шарды стадии VarWar R&D v3 с проверкой ключей и конфигурации.",
        epilog=(
            "Пример: python examples/merge_rnd_shards.py results/rnd_v3_1/shards/* "
            "--stage screening --output results/rnd_v3_1/screening"
        ),
    )
    parser.add_argument(
        "shards",
        nargs="+",
        type=Path,
        help="Каталоги шардов: <stage>-каталог, каталог, содержащий <stage>/, "
        "или корень со старой раскладкой shard_*/<stage>.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Каталог результата.")
    parser.add_argument("--stage", default="screening")
    parser.add_argument(
        "--allow-code-mismatch",
        action="store_true",
        help="Не требовать совпадения версии кода между шардами (не рекомендуется).",
    )
    args = parser.parse_args()
    try:
        merged = merge_shards(
            args.shards,
            args.output,
            stage=args.stage,
            allow_code_mismatch=args.allow_code_mismatch,
        )
    except ShardMergeError as error:
        raise SystemExit(f"Merge refused: {error}") from error
    summary = {key: merged[key] for key in ("n_trial_rows", "n_aggregate_rows", "n_worlds", "n_pipelines")}
    summary["finalists"] = merged["finalists"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
