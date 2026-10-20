#!/usr/bin/env bash
# Сборка PDF-отчёта: рисунки → XeLaTeX (tectonic).
# Требуется tectonic (https://tectonic-typesetting.github.io), путь можно задать TECTONIC=...
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../../../.venv/bin/python}"
"$PY" sim_theory.py >/dev/null
"$PY" make_figures.py
"${TECTONIC:-tectonic}" -X compile report.tex
mkdir -p ../../../outputs/reports
cp report.pdf ../../../outputs/reports/VarWar_Lin_HC2_report_ru.pdf
echo "PDF: outputs/reports/VarWar_Lin_HC2_report_ru.pdf"
