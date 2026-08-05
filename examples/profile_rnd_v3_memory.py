from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar.rnd_runner import RNDRunConfig, profile_rnd_pipelines_peak_rss


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile VarWar v3 pipelines by RSS")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    results = args.results_dir.resolve()
    metadata = json.loads((results / "run_metadata.json").read_text(encoding="utf-8"))
    stored = metadata["config"]
    pipeline_ids = tuple(stored.get("pipeline_ids", ()))
    if not pipeline_ids:
        raise ValueError("run_metadata.json does not contain explicit pipeline_ids")

    config = RNDRunConfig(
        stage="smoke",
        world_ids=(stored["world_ids"][0],),
        effects=(0.0, 0.2),
        n_trials=2,
        seed=int(stored["seed"]) + 100_000,
        resamples=int(stored.get("resamples", 99)),
        include_hypex_assignment=False,
    )
    profile = profile_rnd_pipelines_peak_rss(
        config,
        pipeline_ids,
        results / "memory",
        repeats=args.repeats,
    )
    if (profile["exit_code"] != 0).any():
        raise RuntimeError("At least one isolated memory profile failed")
    summary = (
        profile.groupby("pipeline", as_index=False)["peak_rss_mb"]
        .median()
        .rename(columns={"peak_rss_mb": "peak_rss_mb"})
    )

    for stem in ("aggregates", "leaders"):
        parquet = results / f"{stem}.parquet"
        frame = pd.read_parquet(parquet)
        frame = frame.drop(columns=["peak_rss_mb"], errors="ignore").merge(
            summary, on="pipeline", how="left"
        )
        frame.to_parquet(parquet, index=False)
        frame.to_csv(results / f"{stem}.csv", index=False, encoding="utf-8-sig")

    payload_path = results / "excel_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    memory = dict(zip(summary["pipeline"], summary["peak_rss_mb"]))
    for section in ("aggregates", "leaders"):
        for row in payload.get(section, []):
            row["peak_rss_mb"] = memory.get(row.get("pipeline"))
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
