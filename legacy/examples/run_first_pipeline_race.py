from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import sys
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar import (
    SyntheticPipelineRaceConfig,
    format_pipeline_summary_ru,
    profile_synthetic_pipeline_memory,
    run_synthetic_pipeline_race,
    summarize_synthetic_pipeline_race,
)


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _load_config(path: Path, trials_override: int | None) -> tuple[
    SyntheticPipelineRaceConfig,
    Path,
]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    simulation = raw["simulation"]
    strata = raw["stratification"]
    runtime = raw["runtime"]
    config = SyntheticPipelineRaceConfig(
        n_trials=(
            int(trials_override)
            if trials_override is not None
            else int(simulation["n_trials"])
        ),
        n=int(simulation["n"]),
        alternative_effect=float(simulation["alternative_effect"]),
        covariate_signal=float(simulation["covariate_signal"]),
        noise_scale=float(simulation["noise_scale"]),
        heavy_tail=bool(simulation["heavy_tail"]),
        seed=int(simulation["seed"]),
        alpha=float(simulation["alpha"]),
        category_cols=tuple(strata["category_cols"]),
        n_strata=int(strata["n_strata"]),
        population_size=int(strata["population_size"]),
        max_generations=int(strata["max_generations"]),
        elite_size=int(strata["elite_size"]),
        patience=int(strata["patience"]),
        min_macro_stratum_size=int(strata["min_macro_stratum_size"]),
        mutation_probability=float(strata["mutation_probability"]),
        prefer_pypi_hypex=bool(runtime["prefer_pypi_hypex"]),
    )
    return config, ROOT / str(raw["output"]["directory"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the first full-traffic synthetic pipeline race."
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--no-save-raw", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "first_pipeline_race.toml",
    )
    args = parser.parse_args()

    config, output_dir = _load_config(args.config, args.trials)
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress(done: int, total: int) -> None:
        if done == 1 or done == total or done % 10 == 0:
            print(f"Готово trial-пар: {done}/{total}", flush=True)

    raw = run_synthetic_pipeline_race(config, progress=progress)
    summary = summarize_synthetic_pipeline_race(raw, alpha=config.alpha)
    memory_profile = profile_synthetic_pipeline_memory(config)
    summary = summary.merge(
        memory_profile,
        on=["data_scenario", "track", "pipeline"],
        how="left",
        validate="one_to_one",
    )
    russian = format_pipeline_summary_ru(summary)

    summary.to_csv(output_dir / "summary.csv", index=False)
    memory_profile.to_csv(output_dir / "memory_profile.csv", index=False)
    if not args.no_save_raw:
        raw.to_csv(
            output_dir / "raw_results.csv.gz",
            index=False,
            compression="gzip",
        )
    metadata = {
        "config": dataclasses.asdict(config),
        "rows": int(len(raw)),
        "successful_rows": int((raw["status"] == "ok").sum()),
        "backends": sorted(
            str(value)
            for value in raw.get("diag_backend", pd.Series(dtype=object))
            .dropna()
            .unique()
        ),
        "memory_profile": {
            "trials": 1,
            "seed": config.seed,
            "warmup": True,
            "metric": "tracemalloc peak Python allocations",
            "includes_native_numpy_blas_buffers": False,
        },
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("hypex", "numpy", "pandas", "scipy")
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = "\n".join(
        [
            "# Первая синтетическая гонка pipelines",
            "",
            "Все строки используют один и тот же полный экспериментальный трафик. "
            "H0 и H1 внутри trial имеют одинаковые признаки, noise и assignment; "
            "различается только добавленный treatment effect.",
            "",
            f"- Повторов для каждой гипотезы: `{config.n_trials}`.",
            f"- Наблюдений в эксперименте: `{config.n}`.",
            f"- Истинный эффект H1: `{config.alternative_effect}`.",
            f"- Уровень значимости: `{config.alpha}`.",
            "- GA обучается на отдельной historical-выборке по historical target `y`.",
            "- Режим Неймана не участвует; архив находится в `legacy/neyman/`.",
            "",
            "Статус «FPR калиброван» означает, что exact binomial test не отвергает "
            "номинальный FPR 5% на текущем числе повторов.",
            "",
            "## Сводная таблица",
            "",
            _markdown_table(russian),
            "",
            "## Как интерпретировать эту гонку",
            "",
            "- Практический объект выбора — полный pipeline. Сравнение отдельных "
            "компонентов всё равно сохраняется как ablation: оно показывает, за "
            "счёт какого шага появился выигрыш или нарушилась калибровка.",
            "- Ранжировать pipelines только по мощности нельзя. Сначала проверяются "
            "FPR, покрытие доверительного интервала и смещение, затем дисперсия, "
            "мощность, runtime и память.",
            "- Колонка памяти показывает end-to-end пик Python-аллокаций в "
            "отдельном steady-state profile-прогоне с warm-up и тем же seed. Для "
            "GA это максимум между fit preprocessing и estimator; native-буферы "
            "NumPy/BLAS в метрику не входят. Отдельный прогон не искажает основной "
            "runtime.",
            "- Welch t-test — канонический статистический baseline. HypEx A/B — "
            "интеграционный baseline: при одинаковом estimand их статистические "
            "результаты должны совпадать с точностью вычислений.",
            "- Эта первая гонка покрывает один благоприятный линейный сценарий с "
            "нормальным шумом. Она отбирает кандидатов для следующих сценариев, "
            "но не устанавливает универсального победителя.",
            "- При 100 повторах интервалы для FPR и мощности ещё широки. Для "
            "финального решения нужны дополнительные сценарии и больше повторов.",
            "",
        ]
    )
    (output_dir / "summary_ru.md").write_text(report, encoding="utf-8")
    print(russian.to_string(index=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
