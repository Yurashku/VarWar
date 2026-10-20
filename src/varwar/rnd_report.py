from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from varwar.validation import QUALITY_ORDER, quality_rank


_DUPLICATE_METRICS = (
    "inference_quality",
    "h0_rejection_rate",
    "h0_truth_rejection_rate",
    "h1_bias",
    "h1_rmse",
    "h1_coverage",
    "h1_se_to_sd",
    "h1_rejection_rate",
)


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


def _fmt(value: object, spec: str) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    return format(number, spec) if math.isfinite(number) else ""


def collapse_context_duplicates(aggregates: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that differ only by an ignored information/temporal context.

    Rows of the same (pipeline, world, estimand, effect) are merged when they
    carry identical validation metrics; this covers broadcast mode-invariant
    rows and also older runs that executed every context separately.  The
    merged row lists its contexts in ``contexts``.
    """

    if aggregates.empty:
        return aggregates.copy()
    frame = aggregates.copy()
    frame["_context"] = (
        frame["information_mode"].astype(str) + "|" + frame["temporal_access"].astype(str)
    )
    metrics = [column for column in _DUPLICATE_METRICS if column in frame]
    signature = frame[metrics].copy()
    for column in metrics:
        if pd.api.types.is_float_dtype(signature[column]):
            signature[column] = signature[column].round(12)
    keys = [
        column
        for column in ("pipeline", "world_id", "ranking_estimand_id", "effect_size")
        if column in frame
    ]
    frame["_signature"] = pd.util.hash_pandas_object(
        pd.concat([frame[keys], signature], axis=1), index=False
    ).astype("uint64")
    if "executed_context" in frame:
        frame["_canonical"] = (frame["executed_context"].astype(str) == frame["_context"]).astype(int)
    else:
        frame["_canonical"] = 0
    contexts = frame.groupby("_signature", sort=False)["_context"].agg(
        lambda values: ", ".join(dict.fromkeys(values))
    )
    counts = frame.groupby("_signature", sort=False)["_context"].size()
    collapsed = (
        frame.sort_values("_canonical", ascending=False, kind="mergesort")
        .drop_duplicates("_signature", keep="first")
        .copy()
    )
    collapsed["contexts"] = collapsed["_signature"].map(contexts)
    collapsed["n_contexts"] = collapsed["_signature"].map(counts).astype(int)
    order = frame.drop_duplicates("_signature")["_signature"]
    collapsed = collapsed.set_index("_signature").loc[order].reset_index(drop=True)
    return collapsed.drop(columns=["_context", "_canonical"])


def sort_by_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Order good < warning < uncertain < bad, then by RMSE and runtime."""

    if frame.empty:
        return frame
    ordered = frame.copy()
    ordered["_quality_rank"] = quality_rank(ordered["inference_quality"])
    by = ["_quality_rank"] + [
        column for column in ("h1_rmse", "runtime_median_ms") if column in ordered
    ]
    ordered = ordered.sort_values(by, na_position="last", kind="mergesort")
    return ordered.drop(columns="_quality_rank")


def _contexts_label(row: pd.Series) -> str:
    contexts = str(row.get("contexts", "") or "")
    n_contexts = int(row.get("n_contexts", 1) or 1)
    if n_contexts >= 4:
        return "любой (не зависит от контекста)"
    return contexts or f"{row.get('information_mode', '')}|{row.get('temporal_access', '')}"


def _size_label(row: pd.Series) -> str:
    if "size" in row and pd.notna(row.get("size")):
        value = _fmt(row["size"], ".3f")
        if row.get("h0_is_null") is False or row.get("h0_is_null") == 0:
            return value + "*"
        return value
    return _fmt(row.get("h0_rejection_rate"), ".3f")


def _summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    def column(name: str) -> pd.Series:
        return frame[name] if name in frame else pd.Series(math.nan, index=frame.index)

    return pd.DataFrame(
        {
            "Пайплайн": frame["pipeline"],
            "Мир": frame["world_id"],
            "Estimand": frame["ranking_estimand_id"],
            "Контексты": frame.apply(_contexts_label, axis=1),
            "Эффект, σ": column("effect_size").map(lambda x: _fmt(x, ".2g")),
            "Вывод": frame["inference_quality"],
            "Причина": column("verdict_reasons").fillna(""),
            "Bias": column("h1_bias").map(lambda x: _fmt(x, ".4g")),
            "MC SE bias": column("h1_bias_mc_se").map(lambda x: _fmt(x, ".2g")),
            "RMSE": column("h1_rmse").map(lambda x: _fmt(x, ".4g")),
            "Размер": frame.apply(_size_label, axis=1),
            "Coverage": column("h1_coverage").map(lambda x: _fmt(x, ".3f")),
            "Power": column("power").map(lambda x: _fmt(x, ".3f")),
            "SE/SD": column("h1_se_to_sd").map(lambda x: _fmt(x, ".3f")),
            "MDE80": column("mde_80").map(lambda x: _fmt(x, ".3g")),
            "Пик памяти, МБ": column("peak_memory_mb").map(lambda x: _fmt(x, ".2f")),
            "Время, мс": column("runtime_median_ms").map(lambda x: _fmt(x, ".1f")),
        }
    )


def render_rnd_report_ru(
    aggregates: pd.DataFrame,
    trials: pd.DataFrame,
    pipeline_matrix: pd.DataFrame,
    *,
    stage: str,
) -> str:
    """Сформировать короткий русский отчёт без ложного рейтинга smoke."""

    status = trials.get("status", pd.Series(dtype=str)).value_counts()
    broadcast = (
        int(trials["broadcast"].fillna(False).astype(bool).sum())
        if "broadcast" in trials
        else 0
    )
    lines = [
        "# Отчёт VarWar R&D v3",
        "",
        f"**Этап:** `{stage}`.",
        "",
        "Единица сравнения — полная процедура назначения и анализа. Рейтинг "
        "строится только внутри одного estimand, контекста и размера эффекта и "
        "только после численного экрана статистического вывода.",
        "",
        "## Полнота запуска",
        "",
        f"- Строк в полной матрице совместимости: **{len(pipeline_matrix):,}**.",
        f"- Trial-level строк: **{len(trials):,}**, из них скопировано из "
        f"исполнения в другом контексте (mode-invariant): **{broadcast:,}**.",
        f"- Успешно: **{int(status.get('ran', 0)):,}**; только точка: "
        f"**{int(status.get('point_only', 0)):,}**; ошибок: "
        f"**{int(status.get('failed', 0)):,}**.",
        "",
        "## Как проверяется статистический вывод",
        "",
        "Размер теста измеряется относительно истины: если под H0 истинный "
        "estimand тождественно равен нулю, это доля p ≤ α; иначе (например, "
        "сдвиг когорты при гетерогенном эффекте) — доля CI, не покрывших "
        "истину, и такая строка помечена `h0_is_null=False` и звёздочкой в "
        "колонке «Размер». Вердикт `bad` ставится только при Монте-Карло "
        "доказательстве провала: нижняя граница Уилсона размера > 0.07, "
        "верхняя граница покрытия < 0.93, |bias| − 1.96·MC SE > 0.10·SD, "
        "верхняя граница CI для SE/SD < 0.90 или отказ super-uniformity после "
        "BH по всей стадии. Превышение точечного порога bias без такой "
        "уверенности даёт `uncertain`; выход точечных FPR/coverage за "
        "[0.03; 0.07]/[0.93; 0.97] — `warning`. Порядок: "
        + " < ".join(f"`{name}`" for name in QUALITY_ORDER[:4])
        + ".",
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
            "- Эффекты разных размеров используют одни и те же реализации (CRN), "
            "поэтому строки δ=0.1 и δ=0.2 не являются независимыми подтверждениями.",
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
    if not aggregates.empty and "inference_quality" in aggregates:
        collapsed = sort_by_quality(collapse_context_duplicates(aggregates))
        counts = collapsed["inference_quality"].value_counts()
        lines.extend(
            [
                "## Сводная таблица",
                "",
                f"Уникальных строк после схлопывания контекстов: **{len(collapsed):,}** "
                f"(исходно {len(aggregates):,}). Вердикты: "
                + ", ".join(
                    f"`{name}` — {int(counts.get(name, 0))}"
                    for name in QUALITY_ORDER
                    if int(counts.get(name, 0))
                )
                + ".",
                "",
                _md_table(_summary_table(collapsed.head(30))),
                "",
                "«Размер»: доля отклонений верной H0; `*` — H0 не нулевая, размер "
                "посчитан как 1 − покрытие истины. «Пик памяти» — пик tracemalloc "
                "аллокаций Python/NumPy внутри одного вызова estimate+inference, "
                "а не RSS процесса.",
                "",
            ]
        )
        problems = collapsed.loc[collapsed["inference_quality"].isin(["bad", UNCERTAIN_LABEL])]
        if "raw_inference_quality" in collapsed:
            problems = collapsed.loc[
                collapsed["raw_inference_quality"].isin(["bad", UNCERTAIN_LABEL])
            ]
        if not problems.empty:
            reasons = problems.copy()
            table = pd.DataFrame(
                {
                    "Пайплайн": reasons["pipeline"],
                    "Мир": reasons["world_id"],
                    "Контексты": reasons.apply(_contexts_label, axis=1),
                    "Эффект, σ": reasons["effect_size"].map(lambda x: _fmt(x, ".2g")),
                    "Вердикт без учёта объёма MC": reasons.get(
                        "raw_inference_quality", reasons["inference_quality"]
                    ),
                    "Причина": reasons.get("verdict_reasons", pd.Series("", index=reasons.index)),
                }
            ).head(30)
            lines.extend(
                [
                    "## Провалы и неопределённости",
                    "",
                    "Причины: `bias`, `fpr`, `coverage`, `se_sd`, "
                    "`pvalue_calibration`, `execution_failure` — доказанный провал; "
                    "`bias_uncertain` — точечный порог превышен, но MC-интервал "
                    "его не исключает.",
                    "",
                    _md_table(table),
                    "",
                ]
            )
        if "failure_rate" in aggregates and (aggregates["failure_rate"].fillna(0) > 0).any():
            failing = collapse_context_duplicates(
                aggregates.loc[aggregates["failure_rate"].fillna(0) > 0]
            )
            lines.extend(
                [
                    "## Строки с ошибками исполнения",
                    "",
                    _md_table(
                        pd.DataFrame(
                            {
                                "Пайплайн": failing["pipeline"],
                                "Мир": failing["world_id"],
                                "Эффект, σ": failing["effect_size"].map(lambda x: _fmt(x, ".2g")),
                                "Доля ошибок": failing["failure_rate"].map(lambda x: _fmt(x, ".3f")),
                            }
                        ).head(30)
                    ),
                    "",
                ]
            )
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
            "1. Совпадают estimand, режим информации, временной доступ и размер эффекта.",
            "2. Лидеры — только строки `good` (флаг `conservative` допускается): "
            "нет доказанного провала, bias, размер, покрытие и SE/SD в целевых зонах.",
            "3. Среди прошедших сравниваются RMSE и power; MDE80, время и пик "
            "памяти — описательные метрики.",
            "4. Финалисты confirmation выбираются кодом "
            "(`select_confirmation_finalists`); список и правило записаны в "
            "`run_metadata.json`.",
            "5. Full-timeline и post-treatment-risk кандидаты ранжируются отдельно от causal-safe.",
            "",
        ]
    )
    return "\n".join(lines)


UNCERTAIN_LABEL = "uncertain"


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
