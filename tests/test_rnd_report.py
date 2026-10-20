from __future__ import annotations

import unittest

import pandas as pd

from varwar.rnd_report import (
    collapse_context_duplicates,
    render_rnd_report_ru,
    sort_by_quality,
)


CONTEXTS = [
    ("experiment_only", "pre_treatment_only"),
    ("experiment_only", "full_timeline"),
    ("panel_prior", "pre_treatment_only"),
    ("panel_prior", "full_timeline"),
]


def _row(pipeline: str, quality: str, rmse: float, info: str, temporal: str, **extra: object) -> dict:
    return {
        "pipeline": pipeline,
        "world_id": "w",
        "ranking_estimand_id": "sample_ate",
        "information_mode": info,
        "temporal_access": temporal,
        "effect_size": 0.2,
        "inference_quality": quality,
        "raw_inference_quality": quality,
        "verdict_reasons": extra.pop("reasons", ""),
        "h0_rejection_rate": 0.05,
        "h1_bias": 0.0,
        "h1_bias_mc_se": 0.01,
        "h1_rmse": rmse,
        "h1_coverage": 0.95,
        "h1_se_to_sd": 1.0,
        "h1_rejection_rate": 0.5,
        "power": 0.5,
        "size": 0.05,
        "h0_is_null": True,
        "mde_80": 0.3,
        "peak_memory_mb": 0.1,
        "runtime_median_ms": 1.0,
        "mode_invariant": True,
        "executed_context": "experiment_only|pre_treatment_only",
        **extra,
    }


def _aggregates() -> pd.DataFrame:
    rows = []
    for info, temporal in CONTEXTS:
        # bad has the lowest RMSE: a lexicographic sort would put it first.
        rows.append(_row("bad_pipe", "bad", 0.05, info, temporal, reasons="bias;fpr", h1_bias=0.3))
        rows.append(_row("good_pipe", "good", 0.10, info, temporal))
        rows.append(_row("warn_pipe", "warning", 0.07, info, temporal, reasons="fpr_low"))
        rows.append(_row("unc_pipe", "uncertain", 0.06, info, temporal, reasons="bias_uncertain"))
    return pd.DataFrame(rows)


def _table_rows(report: str, heading: str) -> list[list[str]]:
    lines = report.splitlines()
    section = []
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            section.append(line)
    return [[cell.strip() for cell in line.strip("|").split(" | ")] for line in section[2:]]


class RNDReportTests(unittest.TestCase):
    def test_quality_is_ordered_categorically(self) -> None:
        ordered = sort_by_quality(_aggregates())
        self.assertEqual(
            ordered["inference_quality"].drop_duplicates().tolist(),
            ["good", "warning", "uncertain", "bad"],
        )

    def test_mode_invariant_contexts_are_collapsed(self) -> None:
        collapsed = collapse_context_duplicates(_aggregates())
        self.assertEqual(len(collapsed), 4)
        self.assertTrue(collapsed["n_contexts"].eq(4).all())
        self.assertTrue(
            collapsed["information_mode"].eq("experiment_only").all()
            and collapsed["temporal_access"].eq("pre_treatment_only").all()
        )

    def test_context_sensitive_rows_are_not_collapsed(self) -> None:
        frame = _aggregates()
        sensitive = frame["pipeline"] == "good_pipe"
        frame.loc[sensitive, "h1_rmse"] = [0.10, 0.11, 0.10, 0.11]
        collapsed = collapse_context_duplicates(frame)
        self.assertEqual(int((collapsed["pipeline"] == "good_pipe").sum()), 2)

    def test_first_summary_row_is_good_and_reasons_are_printed(self) -> None:
        report = render_rnd_report_ru(
            _aggregates(),
            pd.DataFrame({"status": ["ran"] * 4, "pipeline": ["p"] * 4, "world_id": ["w"] * 4}),
            pd.DataFrame({"status": ["eligible"]}),
            stage="screening",
        )
        rows = _table_rows(report, "## Сводная таблица")
        header = [line for line in report.splitlines() if line.startswith("| Пайплайн")][0]
        columns = [cell.strip() for cell in header.strip("|").split(" | ")]
        quality = columns.index("Вывод")
        self.assertEqual(rows[0][quality], "good")
        self.assertEqual(rows[0][0], "good_pipe")
        self.assertEqual(len(rows), 4)
        self.assertIn("## Провалы и неопределённости", report)
        self.assertIn("bias;fpr", report)
        self.assertIn("bias_uncertain", report)
        self.assertIn("любой (не зависит от контекста)", report)


if __name__ == "__main__":
    unittest.main()
