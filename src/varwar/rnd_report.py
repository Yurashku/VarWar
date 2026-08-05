from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Нет строк для отображения._"
    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join((header, separator, *rows))


def render_rnd_report_ru(
    aggregates: pd.DataFrame,
    trials: pd.DataFrame,
    pipeline_matrix: pd.DataFrame,
    *,
    stage: str,
) -> str:
    """Сформировать короткий русский отчёт без ложного рейтинга smoke."""

    status = trials.get("status", pd.Series(dtype=str)).value_counts()
    lines = [
        "# Отчёт VarWar R&D v3",
        "",
        f"**Этап:** `{stage}`.",
        "",
        "Единица сравнения — полная процедура назначения и анализа. Рейтинг "
        "строится только внутри одного estimand и только после численного "
        "экрана статистического вывода.",
        "",
        "## Полнота запуска",
        "",
        f"- Строк в полной матрице совместимости: **{len(pipeline_matrix):,}**.",
        f"- Исполненных trial-level строк: **{len(trials):,}**.",
        f"- Успешно: **{int(status.get('ran', 0)):,}**; только точка: "
        f"**{int(status.get('point_only', 0)):,}**; ошибок: "
        f"**{int(status.get('failed', 0)):,}**.",
        "",
        "## Как проверяется статистический вывод",
        "",
        "Под H0 измеряются FPR с интервалом Уилсона и super-uniformity "
        "p-value. Под H1 измеряются bias, RMSE, empirical SD, средний SE, "
        "SE/SD, покрытие CI, power и ошибка второго рода. Значение FPR=0 на "
        "малом числе повторов помечается как неопределённое, а не идеальное.",
        "",
    ]
    executed_pipelines = int(trials["pipeline"].nunique()) if "pipeline" in trials else 0
    executed_worlds = int(trials["world_id"].nunique()) if "world_id" in trials else 0
    monte_carlo_sizes = sorted(
        int(value)
        for value in aggregates.get("h0_n_trials", pd.Series(dtype=int)).dropna().unique()
    )
    lines.extend(
        [
            "## Границы численного результата",
            "",
            f"- Реально исполнено уникальных процедур: **{executed_pipelines}**.",
            f"- Реально исполнено синтетических миров: **{executed_worlds}**.",
            f"- Число H0-реализаций на сравнимую строку: **{monte_carlo_sizes or 'нет'}**.",
            "- Полная матрица совместимости шире исполненного этапа; "
            "неисполненная строка не считается проверенным методом.",
            "",
        ]
    )
    if stage == "smoke":
        lines.extend(
            [
                "> Smoke нужен для проверки исполнимости. Пять повторений не "
                "дают научного рейтинга и не позволяют заключать о FPR или power.",
                "",
            ]
        )
    if not aggregates.empty:
        compact = aggregates.copy()
        compact = compact.sort_values(
            ["inference_quality", "h1_rmse", "runtime_median_ms"],
            na_position="last",
        ).head(30)
        table = pd.DataFrame(
            {
                "Пайплайн": compact["pipeline"],
                "Мир": compact["world_id"],
                "Estimand": compact["ranking_estimand_id"],
                "Информация": compact["information_mode"],
                "Время": compact["temporal_access"],
                "Эффект, σ": compact["effect_size"].map(lambda x: f"{x:.2g}"),
                "Вывод": compact["inference_quality"],
                "Bias": compact["h1_bias"].map(lambda x: f"{x:.4g}"),
                "RMSE": compact["h1_rmse"].map(lambda x: f"{x:.4g}"),
                "FPR": compact["h0_rejection_rate"].map(lambda x: f"{x:.3f}"),
                "Coverage": compact["h1_coverage"].map(lambda x: f"{x:.3f}"),
                "Power": compact["power"].map(lambda x: f"{x:.3f}"),
                "SE/SD": compact["h1_se_to_sd"].map(lambda x: f"{x:.3f}"),
                "RSS, MB": compact.get("peak_rss_mb", pd.Series(index=compact.index)).map(
                    lambda x: f"{x:.1f}" if pd.notna(x) else ""
                ),
            }
        )
        lines.extend(["## Сводная таблица", "", _md_table(table), ""])
    failures = trials.loc[trials.get("status") == "failed"] if not trials.empty else trials
    if not failures.empty:
        reasons = (
            failures["status_reason"].fillna("unknown").value_counts().head(15).rename_axis("Причина").reset_index(name="Строк")
        )
        lines.extend(["## Ошибки исполнения", "", _md_table(reasons), ""])
    lines.extend(
        [
            "## Правило выбора",
            "",
            "1. Совпадают estimand, режим информации и временной доступ.",
            "2. Проходят bias, FPR, p-value, coverage и SE/SD.",
            "3. Среди прошедших сравниваются RMSE, variance reduction, power, MDE, время и RSS.",
            "4. Full-timeline и post-treatment-risk кандидаты ранжируются отдельно от causal-safe.",
            "",
        ]
    )
    return "\n".join(lines)


def write_rnd_report_ru(
    path: str | Path,
    aggregates: pd.DataFrame,
    trials: pd.DataFrame,
    pipeline_matrix: pd.DataFrame,
    *,
    stage: str,
) -> Path:
    target = Path(path)
    target.write_text(
        render_rnd_report_ru(
            aggregates, trials, pipeline_matrix, stage=stage
        ),
        encoding="utf-8",
    )
    return target
