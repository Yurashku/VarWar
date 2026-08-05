from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar.method_catalog import method_catalog
from varwar.rnd_report import write_rnd_report_ru
from varwar.rnd_runner import aggregate_rnd_results, select_rnd_leaders
from varwar.synthetic_v3 import list_worlds


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed VarWar R&D shards.")
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stage", default="screening")
    args = parser.parse_args()

    shard_dirs = sorted(args.shard_root.glob(f"shard_*/{args.stage}"))
    if not shard_dirs:
        raise SystemExit("No completed shard directories found.")
    missing = [path for path in shard_dirs if not (path / "trial_results.parquet").exists()]
    if missing:
        raise SystemExit(f"Incomplete shards: {missing}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    trials = pd.concat(
        [pd.read_parquet(path / "trial_results.parquet") for path in shard_dirs],
        ignore_index=True,
    )
    matrix = pd.read_parquet(shard_dirs[0] / "pipeline_matrix.parquet")
    aggregates = aggregate_rnd_results(trials)
    leaders = select_rnd_leaders(aggregates)

    trials.to_parquet(output / "trial_results.parquet", index=False)
    # The compatibility matrix is identical in every shard. Copying the
    # original Parquet also avoids an expensive re-encoding of 762k rows.
    shutil.copyfile(
        shard_dirs[0] / "pipeline_matrix.parquet",
        output / "pipeline_matrix.parquet",
    )
    aggregates.to_parquet(output / "aggregates.parquet", index=False)
    leaders.to_parquet(output / "leaders.parquet", index=False)
    aggregates.to_csv(output / "aggregates.csv", index=False, encoding="utf-8-sig")
    leaders.to_csv(output / "leaders.csv", index=False, encoding="utf-8-sig")

    runnable = matrix.loc[matrix["status"].isin(["eligible", "point_only"])]
    methods = pd.DataFrame(
        {
            "method_id": record.method_id,
            "category": record.category.value,
            "status": record.status.value,
            "summary": record.summary,
            "implementation": record.implementation,
            "race_scope": record.race_scope,
        }
        for record in method_catalog()
        if record.category.value != "CompositePipeline"
    )
    payload = {
        "aggregates": json.loads(aggregates.to_json(orient="records")),
        "leaders": json.loads(leaders.to_json(orient="records")),
        "pipelines": json.loads(runnable.to_json(orient="records")),
        "methods": json.loads(methods.to_json(orient="records")),
        "worlds": json.loads(list_worlds().to_json(orient="records")),
    }
    (output / "excel_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "stage": args.stage,
        "shards": [str(path) for path in shard_dirs],
        "n_matrix_rows": len(matrix),
        "n_trial_rows": len(trials),
        "n_aggregate_rows": len(aggregates),
        "n_worlds": int(trials["world_id"].nunique()),
        "n_pipelines": int(trials["pipeline"].nunique()),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_rnd_report_ru(
        output / "report_ru.md", aggregates, trials, matrix, stage=args.stage
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
