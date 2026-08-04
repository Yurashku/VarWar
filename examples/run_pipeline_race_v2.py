from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import sys
import tempfile
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from varwar.pipeline_benchmark import (
    SCENARIOS,
    SyntheticPipelineRaceConfig,
    format_pipeline_summary_ru,
    pipeline_definitions,
    profile_synthetic_pipeline_memory,
    run_synthetic_pipeline_race,
    run_synthetic_pipeline_race_parallel,
    select_best_pipelines,
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


def _load_config(
    path: Path,
    trials_override: int | None,
) -> tuple[SyntheticPipelineRaceConfig, Path]:
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
        seed=int(simulation["seed"]),
        alpha=float(simulation["alpha"]),
        scenario_ids=tuple(simulation["scenario_ids"]),
        category_cols=tuple(strata["category_cols"]),
        n_strata=int(strata["n_strata"]),
        population_size=int(strata["population_size"]),
        max_generations=int(strata["max_generations"]),
        elite_size=int(strata["elite_size"]),
        patience=int(strata["patience"]),
        min_macro_stratum_size=int(strata["min_macro_stratum_size"]),
        mutation_probability=float(strata["mutation_probability"]),
        prefer_pypi_hypex=bool(runtime["prefer_pypi_hypex"]),
        include_stratified=bool(strata["enabled"]),
        information_mode=str(simulation.get("information_mode", "both")),
        panel_prior_months=int(simulation.get("panel_prior_months", 6)),
        include_non_rct_specialized=bool(
            simulation.get("include_non_rct_specialized", False)
        ),
        include_cartesian=bool(simulation.get("include_cartesian", False)),
    )
    return config, ROOT / str(raw["output"]["directory"])


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _validate_race_matrix(
    raw: pd.DataFrame,
    config: SyntheticPipelineRaceConfig,
) -> None:
    expected_pipelines = {
        definition.pipeline_id for definition in pipeline_definitions(config)
    }
    expected_scenarios = (
        {"linear_heavy_tail" if config.heavy_tail else "linear_gaussian"}
        if config.heavy_tail is not None
        else set(config.scenario_ids)
    )
    modes = (
        ("experiment_only", "panel_prior")
        if config.information_mode == "both"
        else (config.information_mode,)
    )
    population_ids = {
        "stratified_mean_diff",
        "population_weighted_stratified",
        "calibration_reweighting",
    }
    catalog_definitions = pipeline_definitions(config)
    base_definitions = [
        definition
        for definition in catalog_definitions
        if definition.pipeline_id not in population_ids
    ]
    population_definitions = [
        definition
        for definition in catalog_definitions
        if definition.pipeline_id in population_ids
    ]
    expected_by_mode = {}
    for mode in modes:
        definitions_for_mode = base_definitions
        if mode == "panel_prior" and "population_shifted_strata" in expected_scenarios:
            definitions_for_mode = base_definitions + population_definitions
        expected_by_mode[mode] = {
            definition.pipeline_id
            for definition in definitions_for_mode
            if definition.information_mode == "experiment_only"
            or definition.information_mode == mode
        }
    expected_rows = 0
    for scenario_id in expected_scenarios:
        for mode in modes:
            scenario_definitions = base_definitions
            if mode == "panel_prior" and scenario_id == "population_shifted_strata":
                scenario_definitions = base_definitions + population_definitions
            expected_rows += config.n_trials * 2 * sum(
                definition.information_mode == "experiment_only"
                or definition.information_mode == mode
                for definition in scenario_definitions
            )
    if len(raw) != expected_rows:
        raise RuntimeError(
            f"Race matrix has {len(raw)} rows; expected {expected_rows}."
        )
    if set(raw["pipeline"]) != expected_pipelines:
        raise RuntimeError("Race matrix does not contain the expected pipelines.")
    if set(raw["data_scenario"]) != expected_scenarios:
        raise RuntimeError("Race matrix does not contain the expected scenarios.")
    if set(raw["hypothesis"]) != {"H0", "H1"}:
        raise RuntimeError("Race matrix must contain both H0 and H1.")
    observed_by_mode = {
        mode: set(raw.loc[raw["race_information_mode"] == mode, "pipeline"])
        for mode in modes
    }
    if observed_by_mode != expected_by_mode:
        raise RuntimeError(
            "Race matrix does not contain the expected pipelines by information mode."
        )
    keys = [
        "data_scenario",
        "race_information_mode",
        "pipeline",
        "hypothesis",
        "trial",
    ]
    if raw.duplicated(keys).any():
        raise RuntimeError("Race matrix contains duplicate scenario/pipeline trials.")


def _plot_scenarios(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    try:
        cache_dir = Path(tempfile.gettempdir()) / "varwar-matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: list[Path] = []
    for scenario_id, frame in summary.groupby("data_scenario", sort=False):
        comparable = frame.loc[
            (frame["pipeline_tier"] != "changed_metric")
            & frame["ranking_eligible"]
        ].sort_values("empirical_sd_h1", ascending=True)
        if comparable.empty:
            continue
        height = max(5.0, 0.38 * len(comparable))
        figure, axis = plt.subplots(figsize=(11, height))
        positions = range(len(comparable))
        colors = [
            "#18864b" if valid else "#bd3b32"
            for valid in comparable["passes_diagnostic_screen"]
        ]
        axis.errorbar(
            comparable["mean_effect_h1"],
            list(positions),
            xerr=1.96 * comparable["mean_se_h1"],
            fmt="none",
            ecolor="#7b8790",
            elinewidth=1.2,
            capsize=3,
            zorder=1,
        )
        axis.scatter(
            comparable["mean_effect_h1"],
            list(positions),
            c=colors,
            s=34,
            zorder=2,
        )
        axis.axvline(
            float(comparable["true_effect"].iloc[0]),
            color="#111111",
            linestyle="--",
            linewidth=1.2,
            label="Истинный ATE",
        )
        axis.set_yticks(list(positions))
        axis.set_yticklabels(comparable["pipeline_name_ru"], fontsize=8)
        axis.set_xlabel("Средняя оценка эффекта и средняя заявленная 95% ДИ")
        axis.set_title(str(comparable["scenario_name_ru"].iloc[0]))
        axis.grid(axis="x", alpha=0.22)
        axis.legend(loc="best")
        figure.tight_layout()
        path = output_dir / f"effect_ci_{scenario_id}.png"
        figure.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _winner_sentence(frame: pd.DataFrame) -> str:
    selected = select_best_pipelines(frame)
    eligible = selected.loc[selected["selected_pipeline"].notna()]
    if eligible.empty:
        if "scientific_sample_size_ok" in frame.columns and not bool(
            frame["scientific_sample_size_ok"].all()
        ):
            return (
                "Отбор не выполнен: недостаточно повторений для научного рейтинга. "
                "Результаты оставлены диагностическими."
            )
        return "Отбор не дал кандидата после заданного диагностического экрана."
    if len(eligible) > 1:
        return (
            "Есть выбранные пайплайны в нескольких информационных и временных "
            "контурах; см. отдельные строки отбора ниже."
        )
    winner = eligible.iloc[0]
    reduction = 100.0 * float(winner["selected_variance_reduction_vs_welch"])
    return (
        f"Выбран пайплайн **{winner['selected_pipeline_name_ru']}** по критерию "
        f"минимального эмпирического SD после диагностического экрана; "
        f"снижение дисперсии относительно Welch: **{reduction:.1f}%**, "
        f"SD: **{float(winner['selected_empirical_sd_h1']):.4f}**, "
        f"время: **{float(winner['selected_median_runtime_ms']):.2f} мс**. "
        "Это численное ранжирование, а не доказательство превосходства "
        "над близкими кандидатами."
    )


def _compact_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Пайплайн": frame["pipeline_name_ru"],
            "Режим информации": frame.get(
                "race_information_mode", pd.Series("experiment_only", index=frame.index)
            ),
            "Временной доступ": frame.get(
                "temporal_mode", pd.Series("pre_treatment_only", index=frame.index)
            ),
            "Estimand": frame.get("estimand_id", frame["estimand"]),
            "Повторов": frame["n_trials"],
            "Контур": frame["pipeline_tier"],
            "Диагностический статус": frame["diagnostic_status"],
            "Допущен в рейтинг": frame.get(
                "ranking_eligible", pd.Series(False, index=frame.index)
            ),
            "Bias": frame["bias_h1"].round(4),
            "SD": frame["empirical_sd_h1"].round(4),
            "SE/SD": frame["se_to_empirical_sd"].round(3),
            "VR к Welch, %": (
                100.0 * frame["variance_reduction_vs_welch"]
            ).round(1),
            "Инкремент, %": (
                100.0 * frame["incremental_variance_reduction"]
            ).round(1),
            "FPR, %": (100.0 * frame["false_positive_rate"]).round(1),
            "Power, %": (100.0 * frame["power"]).round(1),
            "Coverage, %": (100.0 * frame["coverage_h1"]).round(1),
            "Время, мс": frame["median_runtime_ms"].round(2),
        }
    )


def _render_report(
    summary: pd.DataFrame,
    catalog: pd.DataFrame,
    config: SyntheticPipelineRaceConfig,
    plot_paths: list[Path],
) -> str:
    plot_by_scenario = {
        path.stem.removeprefix("effect_ci_"): path.name for path in plot_paths
    }
    lines = [
        "# Гонка аналитических пайплайнов VarWar v2",
        "",
        "## Короткий вывод",
        "",
        (
            "Сравниваются полные аналитические пайплайны, а не изолированные "
            "примитивы. Сначала проверяются смещение, FPR и покрытие; "
            "только после этого рассматриваются дисперсия, мощность и стоимость."
        ),
        "",
    ]
    for scenario_id in config.scenario_ids:
        frame = summary.loc[summary["data_scenario"] == scenario_id]
        lines.append(
            f"- **{SCENARIOS[scenario_id].name_ru}.** {_winner_sentence(frame)}"
        )

    lines.extend(
        [
            "",
            "## Постановка",
            "",
            f"- Monte Carlo повторов на H0 и H1: `{config.n_trials}`.",
            f"- Строк в одном эксперименте: `{config.n}`.",
            f"- Истинный средний эффект H1: `{config.alternative_effect}`.",
            f"- Уровень значимости: `{config.alpha}`.",
            (
                "- H0 и H1 внутри повтора имеют одинаковые признаки, assignment "
                "и шум; отличается только добавленный treatment effect."
            ),
            (
                "- Отдельной обучающей выборки нет: модель CUPAC A использует "
                "`lag2 -> lag1`, локальные CUPED/B/C оценивают коэффициенты "
                "по текущему контролю, а GA использует только pre-target тех "
                "же строк. Интеграционные HypEx CUPED/CUPAC сохраняют политику "
                "пакета и оценивают theta на всём текущем эксперименте."
            ),
            (
                "- A — предэкспериментные признаки; B — экспертно безопасные "
                "признаки текущего периода и отдельная линейная корректировка; "
                "C — ещё одна линейная корректировка после смыслового отсева и "
                "проверок баланса, пропусков и распределения."
            ),
            (
                "- Balance gate не доказывает причинную безопасность. Сценарий "
                "`misclassified_b` намеренно проверяет цену экспертной ошибки."
            ),
            "",
            "## Правило допуска",
            "",
            (
                "Для первичного R&D-экрана заранее заданы практические допуски: "
                "|bias| ≤ 0,02, FPR ≤ 8%, покрытие ≥ 92% и SE/SD от 0,80 до "
                "1,20. Также требуются 100% успешных запусков."
            ),
            (
                "Прохождение экрана не доказывает статистическую валидность и "
                "не является результатом множественной проверки. Точные "
                "биномиальные интервалы FPR и power приведены для оценки "
                "неопределённости Monte Carlo."
            ),
            (
                "Винзоризация вынесена в отдельный estimand и не участвует в "
                "рейтинге исходного ATE."
            ),
            "",
            "## Каталог гонки",
            "",
            _markdown_table(
                catalog[
                    [
                        "pipeline",
                        "pipeline_name_ru",
                        "components",
                        "estimand",
                        "pipeline_tier",
                        "predecessor_pipeline",
                    ]
                ]
            ),
            "",
        ]
    )

    chain_ids = {
        "hypex_ab",
        "cupac_linear_A__hypex_ab",
        "cupac_linear_A_plus_B__hypex_ab",
        "cupac_linear_A_plus_B_plus_C__hypex_ab",
        "hypex_cupac_temporal",
    }
    for scenario_id in config.scenario_ids:
        frame = summary.loc[summary["data_scenario"] == scenario_id]
        lines.extend(
            [
                f"## {SCENARIOS[scenario_id].name_ru}",
                "",
                SCENARIOS[scenario_id].purpose_ru,
                "",
                _winner_sentence(frame),
                "",
                "### Отбор по сопоставимым контурам",
                "",
            ]
        )
        selection = select_best_pipelines(frame)
        lines.append(_markdown_table(selection))
        lines.extend(
            [
                "",
                "### Цепочка A/B/C",
                "",
                _markdown_table(
                    _compact_table(
                        frame.loc[frame["pipeline"].isin(chain_ids)].sort_values(
                            "empirical_sd_h1"
                        )
                    )
                ),
                "",
                "### Все пайплайны",
                "",
                _markdown_table(
                    _compact_table(frame.sort_values("empirical_sd_h1"))
                ),
                "",
            ]
        )
        plot = plot_by_scenario.get(scenario_id)
        if plot:
            lines.extend(
                [
                    f"![Оценки эффекта: {SCENARIOS[scenario_id].name_ru}]({plot})",
                    "",
                ]
            )

    lines.extend(
        [
            "## Проверки архитектуры",
            "",
            (
                "- `HypEx A/B` — интеграционный parity baseline. Его расхождение "
                "с локальным Welch должно быть только вычислительным."
            ),
            (
                "- `HypEx temporal CUPAC` не включён в цепочку A/B/C: это "
                "библиотечная проверка, а не очередной этап локальной схемы."
            ),
            (
                "- `A+B` и `A+B+C` являются отдельными примитивами политики "
                "данных и отдельными полными пайплайнами."
            ),
            (
                "- GA не считается estimator: runtime preprocessing учитывается "
                "в end-to-end времени соответствующих пайплайнов."
            ),
            "",
            "## Оптимальный транспорт",
            "",
            (
                "OT-тест однородности распределений и estimator эффекта через "
                "OT-matching являются разными методами. Поэтому они не смешаны "
                "с этой гонкой ATE. Для OT-matching нужен отдельный "
                "наблюдательный контур с явно выбранными ATT/ATC, правилами "
                "сопоставления массы и статистического вывода."
            ),
            "",
            "## Ограничения",
            "",
            (
                "- Monte Carlo интервалы всё ещё конечной ширины; близкие методы "
                "нельзя объявлять различными без отдельного paired сравнения."
            ),
            (
                "- C-gate проверяет среднее, пропуски и одномерное распределение, "
                "но сценарий загрязнения пока содержит только легко обнаружимый "
                "постоянный сдвиг. Нужны отдельные проверки условных и слабых сдвигов."
            ),
            (
                "- B/C оцениваются по текущему контролю, а обычный Welch считает "
                "преобразованный target фиксированным. До cross-fitting или "
                "специальной поправки дисперсии эти этапы остаются исследовательскими."
            ),
            (
                "- Пик памяти основан на `tracemalloc` и не включает native "
                "буферы NumPy/BLAS/HypEx."
            ),
            (
                "- Синтетический победитель является кандидатом для следующего "
                "этапа, а не универсальным production default."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the VarWar v2 synthetic pipeline race."
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--no-save-raw", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress and the final console table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; relative paths are resolved from the repository.",
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        default=None,
        help="Run only this pipeline ID; repeat the option for several pipelines.",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help=(
            "Add the full Cartesian product of compatible RCT estimators and "
            "target transformations."
        ),
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help=(
            "Replace selected pipeline rows in an existing raw_results.csv.gz "
            "and regenerate the complete report."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent worker processes for Monte Carlo trials.",
    )
    parser.add_argument(
        "--no-plain-raw",
        action="store_true",
        help="Keep only raw_results.csv.gz; omit the redundant uncompressed CSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pipeline_race_v2.toml",
    )
    args = parser.parse_args()

    base_config, output_dir = _load_config(args.config, args.trials)
    if args.merge_existing and not args.pipeline:
        parser.error("--merge-existing requires at least one --pipeline.")
    config = dataclasses.replace(
        base_config,
        pipeline_ids=tuple(args.pipeline) if args.pipeline else base_config.pipeline_ids,
        include_cartesian=(base_config.include_cartesian or args.cartesian),
    )
    pipeline_definitions(config)
    if args.output_dir is not None:
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else ROOT / args.output_dir
        )
    elif args.trials is not None:
        output_dir = ROOT / "results" / f"pipeline_race_v2_smoke_{args.trials}"
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress(done: int, total: int) -> None:
        if done == 1 or done == total or done % 20 == 0:
            print(f"Готово пар scenario/trial: {done}/{total}", flush=True)

    if args.workers > 1:
        refreshed = run_synthetic_pipeline_race_parallel(
            config,
            workers=args.workers,
            chunk_size=1,
        )
    else:
        refreshed = run_synthetic_pipeline_race(
            config,
            progress=None if args.quiet else progress,
        )
    report_config = config
    if args.merge_existing:
        raw_path = output_dir / "raw_results.csv.gz"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Cannot merge without existing raw results: {raw_path}"
            )
        selected = set(config.pipeline_ids or ())
        existing = pd.read_csv(raw_path, low_memory=False)
        raw = pd.concat(
            [
                existing.loc[~existing["pipeline"].isin(selected)],
                refreshed,
            ],
            ignore_index=True,
            sort=False,
        )
        report_config = dataclasses.replace(base_config, pipeline_ids=None)
    else:
        raw = refreshed
    _validate_race_matrix(raw, report_config)

    summary = summarize_synthetic_pipeline_race(
        raw,
        alpha=report_config.alpha,
    )
    selection = select_best_pipelines(summary)
    memory = profile_synthetic_pipeline_memory(config)
    if args.merge_existing:
        memory_path = output_dir / "memory_profile.csv"
        if not memory_path.exists():
            raise FileNotFoundError(
                f"Cannot merge without existing memory profile: {memory_path}"
            )
        selected = set(config.pipeline_ids or ())
        existing_memory = pd.read_csv(memory_path)
        memory = pd.concat(
            [
                existing_memory.loc[
                    ~existing_memory["pipeline"].isin(selected)
                ],
                memory,
            ],
            ignore_index=True,
            sort=False,
        )
        expected_memory = {
            definition.pipeline_id
            for definition in pipeline_definitions(report_config)
        }
        if (
            set(memory["pipeline"]) != expected_memory
            or memory["pipeline"].duplicated().any()
        ):
            raise RuntimeError("Merged memory profile is incomplete or duplicated.")
    memory_by_pipeline = memory.drop(columns=["data_scenario"])
    summary = summary.merge(
        memory_by_pipeline,
        on="pipeline",
        how="left",
        validate="many_to_one",
    )
    russian = format_pipeline_summary_ru(summary)
    catalog = pd.DataFrame(
        definition.to_record()
        for definition in pipeline_definitions(report_config)
    )
    plots = _plot_scenarios(summary, output_dir)

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    russian.to_csv(output_dir / "summary_ru.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(
        output_dir / "selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    catalog.to_csv(
        output_dir / "pipeline_catalog.csv",
        index=False,
        encoding="utf-8-sig",
    )
    memory.to_csv(
        output_dir / "memory_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not args.no_save_raw:
        raw.to_csv(
            output_dir / "raw_results.csv.gz",
            index=False,
            compression="gzip",
        )
        if args.no_plain_raw:
            plain_raw = output_dir / "raw_results.csv"
            if plain_raw.exists():
                plain_raw.unlink()
        else:
            raw.to_csv(
                output_dir / "raw_results.csv",
                index=False,
                encoding="utf-8-sig",
            )
    else:
        stale_raw = output_dir / "raw_results.csv.gz"
        if stale_raw.exists():
            stale_raw.unlink()
        stale_plain_raw = output_dir / "raw_results.csv"
        if stale_plain_raw.exists():
            stale_plain_raw.unlink()
    metadata = {
        "config": dataclasses.asdict(report_config),
        "rows": int(len(raw)),
        "successful_rows": int((raw["status"] == "ok").sum()),
        "raw_results_saved": not args.no_save_raw,
        "raw_results_plain_saved": (
            not args.no_save_raw and not args.no_plain_raw
        ),
        "selection_file": "selection.csv",
        "selection_criterion": (
            "diagnostic screen + scientific sample size; min empirical SD, "
            "then runtime, then pipeline ID"
        ),
        "workers": int(args.workers),
        "refreshed_pipelines": list(config.pipeline_ids or ()),
        "package_versions": {
            package: _package_version(package)
            for package in ("hypex", "numpy", "pandas", "scipy", "matplotlib")
        },
        "memory_metric": (
            "tracemalloc peak Python allocations; native NumPy/BLAS buffers excluded"
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = _render_report(summary, catalog, report_config, plots)
    (output_dir / "report_ru.md").write_text(report, encoding="utf-8")
    if not args.quiet:
        print(russian.to_string(index=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
