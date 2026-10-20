from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np
import pandas as pd
from scipy import stats

from varwar.rnd_runner import RNDRunConfig, trial_seeds


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "merge_rnd_shards", ROOT / "examples" / "merge_rnd_shards.py"
)
merge = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(merge)

PIPELINE = (
    "adapt.identity__structure.none__assign.complete_randomization"
    "__target.identity__analysis.diff_means__welch"
)


def _write_shard(
    directory: Path,
    world_id: str,
    *,
    seed: int = 20261010,
    n_trials: int = 20,
    git_sha: str = "abc",
    seed_override: int | None = None,
) -> Path:
    stage_dir = directory / "screening"
    stage_dir.mkdir(parents=True)
    rng = np.random.default_rng(zlib.crc32(world_id.encode()))
    rows = []
    for effect in (0.0, 0.2):
        for trial in range(n_trials):
            estimate = effect + rng.normal(0.0, 0.1)
            rows.append(
                {
                    "pipeline": PIPELINE,
                    "world_id": world_id,
                    "information_mode": "experiment_only",
                    "temporal_access": "pre_treatment_only",
                    "ranking_estimand_id": "sample_ate",
                    "assignment": "complete_randomization",
                    "structure": "none",
                    "target_transformer": "identity",
                    "analysis_method": "diff_means__welch",
                    "effect_size": effect,
                    "hypothesis": "H0" if effect == 0.0 else "H1",
                    "trial": trial,
                    "seed": seed_override
                    if seed_override is not None
                    else trial_seeds(seed, world_id, trial)["world"],
                    "status": "ran",
                    "effect": estimate,
                    "standard_error": 0.1,
                    "p_value": float(2 * stats.norm.sf(abs(estimate / 0.1))),
                    "ci_low": estimate - 0.196,
                    "ci_high": estimate + 0.196,
                    "n_control": 100,
                    "n_treatment": 100,
                    "true_sample_ate": effect,
                    "true_population_ate": effect,
                    "true_native_estimand": np.nan,
                    "theory_grade": "A",
                    "causal_safety": "safe",
                    "runtime_ms": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame["seed"] = frame["seed"].astype("uint64")
    frame.to_parquet(stage_dir / "trial_results.parquet", index=False)
    config = RNDRunConfig(
        stage="screening",
        world_ids=(world_id,),
        effects=(0.0, 0.2),
        n_trials=n_trials,
        seed=seed,
        pipeline_ids=(PIPELINE,),
    )
    metadata = {
        "config": config.to_record(),
        "provenance": {"code_version": "0.1.0", "git_sha": git_sha, "git_src_diff_sha1": ""},
        "n_matrix_rows": 10,
        "n_trial_rows": len(frame),
    }
    (stage_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


class MergeShardsTests(unittest.TestCase):
    def test_merge_writes_full_config_and_finalists(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            a = _write_shard(base / "a", "gaussian_r2_04_n200")
            b = _write_shard(base / "b", "contaminated")
            merged = merge.merge_shards([a, b], base / "merged", stage="screening", write_report=True)
            stored = json.loads((base / "merged" / "run_metadata.json").read_text())
            self.assertTrue((base / "merged" / "report_ru.md").exists())
        self.assertEqual(merged["n_worlds"], 2)
        self.assertEqual(stored["config"]["seed"], 20261010)
        self.assertEqual(stored["config"]["n_trials"], 20)
        self.assertEqual(stored["config"]["effects"], [0.0, 0.2])
        self.assertEqual(
            stored["config"]["world_ids"], ["gaussian_r2_04_n200", "contaminated"]
        )
        self.assertEqual(stored["provenance"]["git_sha"], "abc")
        self.assertIn("selection_rule", stored)
        self.assertIn("finalists", stored)
        self.assertEqual(len(stored["merged_from_shards"]), 2)

    def test_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            a = _write_shard(base / "a", "contaminated")
            b = _write_shard(base / "b", "contaminated")
            with self.assertRaisesRegex(merge.ShardMergeError, "share a key"):
                merge.merge_shards([a, b], base / "merged", stage="screening", write_report=False)

    def test_config_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            a = _write_shard(base / "a", "gaussian_r2_04_n200", seed=1)
            b = _write_shard(base / "b", "contaminated", seed=2)
            with self.assertRaisesRegex(merge.ShardMergeError, "seed"):
                merge.merge_shards([a, b], base / "merged", stage="screening", write_report=False)

    def test_code_version_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            a = _write_shard(base / "a", "gaussian_r2_04_n200", git_sha="abc")
            b = _write_shard(base / "b", "contaminated", git_sha="def")
            with self.assertRaisesRegex(merge.ShardMergeError, "git_sha"):
                merge.merge_shards([a, b], base / "merged", stage="screening", write_report=False)
            merged = merge.merge_shards(
                [a, b], base / "merged", stage="screening",
                allow_code_mismatch=True, write_report=False,
            )
            self.assertEqual(merged["n_worlds"], 2)

    def test_seed_scheme_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            a = _write_shard(base / "a", "gaussian_r2_04_n200")
            b = _write_shard(base / "b", "contaminated", seed_override=7)
            with self.assertRaisesRegex(merge.ShardMergeError, "seed"):
                merge.merge_shards([a, b], base / "merged", stage="screening", write_report=False)


if __name__ == "__main__":
    unittest.main()
