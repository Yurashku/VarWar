from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar import AtomicStrataSummary, GeneticStrataReducer


def make_reference(n_rows: int, n_atoms: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    atom = rng.integers(0, n_atoms, size=n_rows, dtype=np.int32)
    atom_signal = np.linspace(-2.0, 2.0, n_atoms)
    target = atom_signal[atom] + rng.normal(0.0, 1.0, size=n_rows)
    return pd.DataFrame({"atom": atom, "y_pre": target})


def main() -> None:
    rows: list[dict[str, float | int]] = []
    for n_rows in (20_000, 200_000):
        reference = make_reference(n_rows, n_atoms=80, seed=42)
        started = time.perf_counter()
        summary = AtomicStrataSummary.from_frame(reference, ("atom",), "y_pre")
        aggregation_ms = (time.perf_counter() - started) * 1000.0

        reducer = GeneticStrataReducer(
            category_cols=("atom",),
            reference_target_col="y_pre",
            n_strata=10,
            population_size=16,
            max_generations=20,
            elite_size=3,
            patience=6,
            random_state=42,
        )
        tracemalloc.start()
        started = time.perf_counter()
        reducer.fit_summary(summary)
        ga_ms = (time.perf_counter() - started) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            {
                "n_rows": n_rows,
                "n_atomic_strata": summary.n_atomic_strata,
                "aggregation_ms": aggregation_ms,
                "ga_fit_summary_ms": ga_ms,
                "ga_peak_memory_bytes": peak_bytes,
                "fitness_evaluations": reducer.diagnostics()["fitness_evaluations"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
