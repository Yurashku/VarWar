import fs from "node:fs/promises";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputDir = path.resolve(process.argv[2] ?? "results/rct_pipeline_race");
const outputPath = path.resolve(
  process.argv[3] ?? "outputs/rct-pipeline-race/VarWar_rct_pipeline_race.xlsx",
);
const previewDir = path.resolve(
  process.argv[4] ?? ".workbook_tmp/rct_race_previews",
);
const skipPreviews = process.env.VARWAR_SKIP_PREVIEWS === "1";

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function scalar(value) {
  if (value === undefined || value === "" || value === "NaN") return null;
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) {
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }
  return value;
}

function csvRows(text) {
  const parsed = parseCsv(text.replace(/^\uFEFF/, ""));
  const headers = parsed[0];
  const values = parsed.slice(1).map((row) =>
    headers.map((_, index) => scalar(row[index] ?? "")),
  );
  return { headers, values };
}

function excelColumn(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function indexByHeader(headers) {
  return Object.fromEntries(headers.map((header, index) => [header, index]));
}

function distinct(values) {
  return [...new Set(values.filter((value) => value !== null && value !== ""))]
    .map(String)
    .sort();
}

async function readRawCsv(inputDirectory) {
  try {
    return await fs.readFile(path.join(inputDirectory, "raw_results.csv"), "utf8");
  } catch (error) {
    const compressed = await fs.readFile(
      path.join(inputDirectory, "raw_results.csv.gz"),
    );
    return gunzipSync(compressed).toString("utf8");
  }
}

const rawSource = csvRows(await readRawCsv(inputDir));
// Excel is the interactive reporting surface. Keep the canonical full raw
// CSV beside it, but write only fields used by filters, formulas, and audit
// of one trial to avoid turning sparse diagnostics into millions of objects.
const workbookRawColumnOrder = [
  "estimator", "effect", "standard_error", "statistic", "p_value", "ci_low",
  "ci_high", "n_control", "n_treatment", "status", "runtime_ms", "pipeline",
  "pipeline_name_ru", "components", "estimand", "pipeline_tier", "data_contract",
  "pipeline_notes", "design_mode", "information_mode", "estimand_id",
  "temporal_mode", "data_scenario", "scenario_name_ru", "scenario_purpose_ru",
  "hypothesis", "true_effect", "trial", "seed", "preprocessor",
  "race_information_mode", "preprocessing_runtime_ms", "end_to_end_runtime_ms",
  "analysis_rows", "run_family",
];
const workbookRawHeaders = workbookRawColumnOrder.filter((header) =>
  rawSource.headers.includes(header),
);
const sourceIndices = workbookRawHeaders.map((header) =>
  rawSource.headers.indexOf(header),
);
const raw = {
  headers: workbookRawHeaders,
  values: rawSource.values.map((row) => sourceIndices.map((index) => row[index])),
};
const summaryText = await fs.readFile(path.join(inputDir, "summary.csv"), "utf8");
const summary = csvRows(summaryText);
const selectionText = await fs
  .readFile(path.join(inputDir, "selection.csv"), "utf8")
  .catch(() => "");
const selection = selectionText ? csvRows(selectionText) : { headers: [], values: [] };
const catalogText = await fs.readFile(path.join(inputDir, "pipeline_catalog.csv"), "utf8");
const catalog = csvRows(catalogText);
const rawIndex = indexByHeader(raw.headers);

const effectColumn = rawIndex.effect;
const rawWithHelpers = raw.values.map((row) => {
  const effect = Number(row[effectColumn]);
  const ciLow = Number(row[rawIndex.ci_low]);
  const ciHigh = Number(row[rawIndex.ci_high]);
  const truth = Number(row[rawIndex.true_effect]);
  return [
    ...row,
    Number.isFinite(effect) ? effect * effect : null,
    Number.isFinite(ciLow) && Number.isFinite(ciHigh) && Number.isFinite(truth)
      ? Number(ciLow <= truth && truth <= ciHigh)
      : null,
  ];
});
const rawHeaders = [...raw.headers, "effect_sq", "covered_true_effect"];
const allRaw = [rawHeaders, ...rawWithHelpers];
const rawLastColumn = excelColumn(rawHeaders.length - 1);
const rawLastRow = allRaw.length;
const includeTrialRows = process.env.VARWAR_INCLUDE_TRIALS === "1"
  || rawSource.values.length <= 25_000;

const summaryIndex = indexByHeader(summary.headers);
const observed = new Map();
for (const row of summary.values) {
  const key = [
    row[summaryIndex.pipeline],
    row[summaryIndex.race_information_mode],
    row[summaryIndex.temporal_mode],
    row[summaryIndex.estimand_id],
  ].join("\u0001");
  if (!observed.has(key)) {
    observed.set(key, {
      pipeline: row[summaryIndex.pipeline],
      pipeline_name_ru: row[summaryIndex.pipeline_name_ru],
      pipeline_tier: row[summaryIndex.pipeline_tier],
      race_information_mode: row[summaryIndex.race_information_mode],
      temporal_mode: row[summaryIndex.temporal_mode],
      estimand_id: row[summaryIndex.estimand_id],
    });
  }
}
const dashboardRows = [...observed.values()].sort((a, b) =>
  `${a.race_information_mode}|${a.pipeline}`.localeCompare(
    `${b.race_information_mode}|${b.pipeline}`,
  ),
);

const workbook = Workbook.create();
const dashboard = workbook.worksheets.add("Dashboard");
const fullRuns = workbook.worksheets.add("Полные прогоны");
const aggregates = workbook.worksheets.add("Агрегированные результаты");
const methods = workbook.worksheets.add("Методы");
const pipelines = workbook.worksheets.add("Пайплайны");
const sources = workbook.worksheets.add("Источники");
const lists = workbook.worksheets.add("Справочники");
const selectionSheet = workbook.worksheets.add("Выбор лучшего");

for (const sheet of [dashboard, fullRuns, aggregates, methods, pipelines, sources, lists, selectionSheet]) {
  sheet.showGridLines = false;
}

const navy = "#17324D";
const blue = "#DCEAF7";
const pale = "#F5F8FB";
const orange = "#FCE4D6";
const green = "#E2F0D9";
const border = "#C9D4DF";
const titleFormat = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 15 } };
const headerFormat = {
  fill: blue,
  font: { bold: true, color: navy },
  borders: { preset: "outside", style: "thin", color: border },
  wrapText: true,
};

if (includeTrialRows) {
  fullRuns.getRange(`A1:${rawLastColumn}${rawLastRow}`).values = allRaw;
  fullRuns.getRange(`A1:${rawLastColumn}1`).format = headerFormat;
  fullRuns.freezePanes.freezeRows(1);
  fullRuns.freezePanes.freezeColumns(5);
  fullRuns.tables.add(`A1:${rawLastColumn}${rawLastRow}`, true, "FullRaceResults");
  fullRuns.getRange(`A1:${rawLastColumn}${rawLastRow}`).format.font = { name: "Aptos", size: 9 };
  fullRuns.getRange(`A1:${rawLastColumn}1`).format = headerFormat;
  for (const [header, width] of Object.entries({
    data_scenario: 20,
    race_information_mode: 18,
    temporal_mode: 22,
    pipeline: 34,
    pipeline_name_ru: 42,
    effect: 13,
    standard_error: 13,
    p_value: 12,
    ci_low: 13,
    ci_high: 13,
    status: 11,
  })) {
    const col = excelColumn(rawIndex[header]);
    fullRuns.getRange(`${col}:${col}`).format.columnWidth = width;
  }
  fullRuns.getRange(`A1:${rawLastColumn}${rawLastRow}`).format.rowHeight = 18;
  fullRuns.getRange(`A2:${rawLastColumn}${rawLastRow}`).format.numberFormat = "General";
  for (const header of ["effect", "standard_error", "p_value", "ci_low", "ci_high", "effect_sq"]) {
    const col = excelColumn(rawHeaders.indexOf(header));
    fullRuns.getRange(`${col}2:${col}${rawLastRow}`).format.numberFormat = "0.0000";
  }
} else {
  fullRuns.getRange("A1:F3").values = [
    ["Trial-level строки вынесены из Excel для компактности.", null, null, null, null, null],
    ["Полный raw_results.csv.gz лежит рядом с книгой.", null, null, null, null, null],
    ["Дашборд работает по агрегатам и не требует этого листа.", null, null, null, null, null],
  ];
  fullRuns.getRange("A1:F1").format = headerFormat;
  fullRuns.getRange("A:A").format.columnWidth = 92;
  fullRuns.getRange("A1:F3").format.wrapText = true;
}

const catalogHeaders = catalog.headers;
pipelines.getRange(`A1:${excelColumn(catalogHeaders.length - 1)}${catalog.values.length + 1}`).values = [
  catalogHeaders,
  ...catalog.values,
];
pipelines.getRange(`A1:${excelColumn(catalogHeaders.length - 1)}1`).format = headerFormat;
pipelines.freezePanes.freezeRows(1);
pipelines.tables.add(`A1:${excelColumn(catalogHeaders.length - 1)}${catalog.values.length + 1}`, true, "PipelineCatalog");
pipelines.getRange(`A1:${excelColumn(catalogHeaders.length - 1)}${catalog.values.length + 1}`).format.font = { name: "Aptos", size: 9 };
pipelines.getRange(`A1:${excelColumn(catalogHeaders.length - 1)}1`).format = headerFormat;
pipelines.getRange("A:A").format.columnWidth = 34;
pipelines.getRange("B:B").format.columnWidth = 42;
pipelines.getRange("C:C").format.columnWidth = 32;
pipelines.getRange("D:D").format.columnWidth = 20;
pipelines.getRange("F:F").format.columnWidth = 28;

const summaryLast = excelColumn(summary.headers.length - 1);
aggregates.getRange(`A1:${summaryLast}${summary.values.length + 1}`).values = [summary.headers, ...summary.values];
aggregates.getRange(`A1:${summaryLast}1`).format = headerFormat;
aggregates.freezePanes.freezeRows(1);
aggregates.tables.add(`A1:${summaryLast}${summary.values.length + 1}`, true, "AggregatedResults");
aggregates.getRange(`A1:${summaryLast}${summary.values.length + 1}`).format.font = { name: "Aptos", size: 9 };
aggregates.getRange(`A1:${summaryLast}1`).format = headerFormat;

const methodRows = [
  ["welch", "Оценка эффекта", "Welch t-test", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; основной статистический baseline."],
  ["hypex_ab", "Оценка эффекта", "HypEx A/B", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; интеграционный baseline PyPI HypEx."],
  ["cuped_control__hypex_ab", "Коррекция target + оценка", "CUPED -> HypEx A/B", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; pre-treatment target, theta по контролю."],
  ["cupac_linear_A__hypex_ab", "Коррекция target + оценка", "CUPAC A -> HypEx A/B", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; temporal lag2 -> lag1."],
  ["cupac_linear_A_plus_B__hypex_ab", "Коррекция target + оценка", "CUPAC A+B -> HypEx A/B", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; B использует разрешённые in-time данные."],
  ["cupac_linear_A_plus_B_plus_C__hypex_ab", "Коррекция target + оценка", "CUPAC A+B+C -> HypEx A/B", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; C является отдельной gated-стадией."],
  ["panel_prior_cupac_A__hypex_ab", "Коррекция target + оценка", "Panel-prior CUPAC A", "РКИ", "panel_prior", "sample_ate", "Реализован; prior только из агрегатов исторической панели."],
  ["panel_prior_cupac_A_plus_B__hypex_ab", "Коррекция target + оценка", "Panel-prior CUPAC A+B", "РКИ", "panel_prior", "sample_ate", "Реализован; A и B не объединяются."],
  ["panel_prior_cupac_A_plus_B_plus_C__hypex_ab", "Коррекция target + оценка", "Panel-prior CUPAC A+B+C", "РКИ", "panel_prior", "sample_ate", "Реализован; при mismatch prior может быть отброшен."],
  ["permutation_mean_diff", "Оценка эффекта + рандомизационный вывод", "Studentized permutation", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; отдельный оценщик, не bootstrap."],
  ["bootstrap_mean_diff", "Оценка эффекта + bootstrap inference", "Studentized bootstrap-t", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; отдельный bootstrap-оценщик."],
  ["regression_adjustment", "Регрессионная корректировка", "Lin regression adjustment", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; centered covariates и HC2 inference."],
  ["ml_rate", "ML-корректировка", "Ridge ML-RATE", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; cross-fitting и ridge."],
  ["nearest_neighbor_matching", "Сопоставление", "Nearest-neighbor matching ATE", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; симметричный ATE."],
  ["optimal_transport_matching", "Сопоставление", "Optimal transport matching ATE", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; regularized transport в RCT ATE-режиме."],
  ["hypex_matching", "Сопоставление", "HypEx Matching ATE", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; parity-адаптер PyPI HypEx."],
  ["isolation_forest_winsorization", "Трансформация target", "Isolation Forest winsorization", "РКИ exploratory", "experiment_only", "changed_metric", "Реализован; treatment-blind contamination, изменённая метрика; вне подтверждающего рейтинга."],
  ["isolation_forest_trimming", "Трансформация target", "Isolation Forest trimming", "РКИ exploratory", "experiment_only", "changed_metric", "Реализован; treatment-blind удаление аномалий; вне подтверждающего рейтинга."],
  ["ga_strata__stratified", "Предобработка дизайна + оценка", "GA strata -> stratified", "РКИ", "experiment_only + panel_prior", "sample_ate", "Реализован; GA не является самостоятельным estimator."],
];
methods.getRange(`A1:G${methodRows.length + 1}`).values = [["ID", "Категория", "English name", "Дизайн", "Информационный режим", "Estimand", "Статус / роль"], ...methodRows];
methods.getRange(`A1:G1`).format = headerFormat;
methods.freezePanes.freezeRows(1);
methods.tables.add(`A1:G${methodRows.length + 1}`, true, "ActiveRaceMethods");
methods.getRange("A:A").format.columnWidth = 38;
methods.getRange("B:B").format.columnWidth = 29;
methods.getRange("C:C").format.columnWidth = 34;
methods.getRange("D:F").format.columnWidth = 24;
methods.getRange("G:G").format.columnWidth = 58;
methods.getRange(`A1:G${methodRows.length + 1}`).format.font = { name: "Aptos", size: 9 };
methods.getRange(`A1:G1`).format = headerFormat;

const sourceRows = [
  ["VarWar", "Код и R&D-гонка", "https://github.com/Yurashku/VarWar"],
  ["HypEx", "PyPI-пакет для integration baseline", "https://pypi.org/project/HypEx/"],
  ["Lin (2013)", "Регрессионная корректировка в рандомизированных экспериментах", "https://doi.org/10.1214/12-AOS1073"],
  ["CUPED", "Variance reduction with pre-experiment data", "https://exp-platform.com/Documents/2013-02-CUPED-Improving%20Sensitivity%20of%20Controlled%20Experiments.pdf"],
  ["CUPAC", "ML-предсказатель для снижения дисперсии", "https://arxiv.org/abs/2211.04663"],
  ["Optimal transport", "Идея matching через transport plan", "https://github.com/HDI-lab-HSE/ABTestingHDILab"],
];
sources.getRange(`A1:C${sourceRows.length + 1}`).values = [["Источник", "Назначение", "URL"], ...sourceRows];
sources.getRange("A1:C1").format = headerFormat;
sources.getRange("A:A").format.columnWidth = 24;
sources.getRange("B:B").format.columnWidth = 54;
sources.getRange("C:C").format.columnWidth = 90;
sources.getRange(`A1:C${sourceRows.length + 1}`).format.font = { name: "Aptos", size: 9 };
sources.getRange("A1:C1").format = headerFormat;

if (selection.headers.length > 0) {
  const selectionLast = excelColumn(selection.headers.length - 1);
  selectionSheet.getRange(`A1:${selectionLast}${selection.values.length + 1}`).values = [
    selection.headers,
    ...selection.values,
  ];
  selectionSheet.getRange(`A1:${selectionLast}1`).format = headerFormat;
  selectionSheet.freezePanes.freezeRows(1);
  selectionSheet.tables.add(
    `A1:${selectionLast}${selection.values.length + 1}`,
    true,
    "BestPipelineSelection",
  );
  selectionSheet.getRange(`A1:${selectionLast}${selection.values.length + 1}`).format.font = {
    name: "Aptos",
    size: 9,
  };
  selectionSheet.getRange(`A1:${selectionLast}${selection.values.length + 1}`).format.wrapText = true;
  selectionSheet.getRange(`A1:${selectionLast}${selection.values.length + 1}`).format.rowHeight = 24;
  selectionSheet.getRange("A:A").format.columnWidth = 24;
  selectionSheet.getRange("B:D").format.columnWidth = 22;
  selectionSheet.getRange("G:G").format.columnWidth = 38;
  selectionSheet.getRange("H:H").format.columnWidth = 38;
  selectionSheet.getRange("I:K").format.columnWidth = 20;
  selectionSheet.getRange("L:M").format.columnWidth = 36;
} else {
  selectionSheet.getRange("A1").values = [[
    "selection.csv отсутствует: запуск не содержал агрегированного критерия отбора.",
  ]];
}

const listValues = {
  information: ["Все", ...distinct(summary.values.map((row) => row[summaryIndex.race_information_mode]))],
  temporal: ["Все", ...distinct(summary.values.map((row) => row[summaryIndex.temporal_mode]))],
  scenario: ["Все", ...distinct(summary.values.map((row) => row[summaryIndex.data_scenario]))],
  tier: ["Все", ...distinct(summary.values.map((row) => row[summaryIndex.pipeline_tier]))],
  estimand: ["Все", ...distinct(summary.values.map((row) => row[summaryIndex.estimand_id]))],
  runFamily: summaryIndex.run_family === undefined
    ? ["Все"]
    : ["Все", ...distinct(summary.values.map((row) => row[summaryIndex.run_family]))],
};
const listRows = [
  ["Информационный режим", ...listValues.information],
  ["Временной доступ", ...listValues.temporal],
  ["Сценарий", ...listValues.scenario],
  ["Контур", ...listValues.tier],
  ["Estimand", ...listValues.estimand],
  ["Семейство прогона", ...listValues.runFamily],
];
lists.getRange(`A1:${excelColumn(Math.max(...listRows.map((row) => row.length)) - 1)}${listRows.length}`).values = listRows;
lists.getRange("A1:F1").format = headerFormat;
lists.getRange("A:A").format.columnWidth = 24;

dashboard.getRange("A1:L1").merge();
dashboard.getRange("A1").values = [["VarWar: дашборд гонки пайплайнов РКИ"]];
dashboard.getRange("A1:L1").format = titleFormat;
dashboard.getRange("A2:L2").merge();
dashboard.getRange("A2").values = [[
  `Фильтры агрегируют лист «Агрегированные результаты»; ${includeTrialRows
    ? "компактные trial-level строки находятся на листе «Полные прогоны»"
    : "trial-level строки доступны в полном raw CSV рядом с книгой"}; рейтинг не следует интерпретировать как доказательство при малом числе повторений.`,
]];
dashboard.getRange("A2:L2").format = { fill: pale, font: { italic: true, color: "#44546A" }, wrapText: true };
dashboard.getRange("A3:B8").values = [
  ["Режим информации гонки", "Все"],
  ["Временной доступ", "Все"],
  ["Сценарий", "Все"],
  ["Контур пайплайна", "Все"],
  ["Estimand", "Все"],
  ["Семейство прогона", "Все"],
];
dashboard.getRange("A3:A8").format = { fill: blue, font: { bold: true, color: navy } };
dashboard.getRange("B3:B8").format = { fill: orange, font: { bold: true, color: navy } };
dashboard.getRange("B3").dataValidation = { rule: { type: "list", values: listValues.information } };
dashboard.getRange("B4").dataValidation = { rule: { type: "list", values: listValues.temporal } };
dashboard.getRange("B5").dataValidation = { rule: { type: "list", values: listValues.scenario } };
dashboard.getRange("B6").dataValidation = { rule: { type: "list", values: listValues.tier } };
dashboard.getRange("B7").dataValidation = { rule: { type: "list", values: listValues.estimand } };
dashboard.getRange("B8").dataValidation = { rule: { type: "list", values: listValues.runFamily } };
dashboard.getRange("D3:F7").values = [
  ["Что агрегируется", "H1", "Средняя оценка, SE, SD, покрытие"],
  ["Проверка ошибки I рода", "H0", "Доля p-value < alpha"],
  ["Доступ к будущему", "Фильтр", "post_treatment_allowed не смешивается с pre-only"],
  ["Внешняя информация", "Фильтр", "panel_prior отдельно от experiment_only"],
  ["Источник формул", "aggregates", "Дашборд считает по листу агрегатов; raw CSV хранит trial × hypothesis"],
];
dashboard.getRange("D3:E7").format = { fill: green, font: { bold: true, color: navy } };
dashboard.getRange("F3:F7").format = { fill: pale, wrapText: true };

const tableStart = 10;
const dashboardHeaders = [
  "Пайплайн", "Название", "Инфо-режим строки", "Временной доступ", "Контур", "Estimand",
  "Число H1", "Средняя оценка H1", "Средняя SE H1", "SD оценок H1", "Покрытие H1", "FPR H0", "Мощность H1", "Среднее время, мс",
];
dashboard.getRange(`A${tableStart}:N${tableStart}`).values = [dashboardHeaders];
dashboard.getRange(`A${tableStart}:N${tableStart}`).format = headerFormat;
const dashboardData = dashboardRows.map((row) => [
  row.pipeline,
  row.pipeline_name_ru,
  row.race_information_mode,
  row.temporal_mode,
  row.pipeline_tier,
  row.estimand_id,
  null, null, null, null, null, null, null, null,
]);
const dashboardEnd = tableStart + dashboardData.length;
dashboard.getRange(`A${tableStart + 1}:N${dashboardEnd}`).values = dashboardData;
// The dashboard reads sufficient aggregates, not trial-level rows. This keeps
// filtering responsive and leaves the full raw CSV as the canonical detail.
const summarySheet = "'Агрегированные результаты'";
const sr = (header) => `${summarySheet}!$${excelColumn(summaryIndex[header])}$2:$${excelColumn(summaryIndex[header])}$${summary.values.length + 1}`;
for (let rowNumber = tableStart + 1; rowNumber <= dashboardEnd; rowNumber += 1) {
  const criteria = [
    `${sr("pipeline")},$A${rowNumber}`,
    `${sr("race_information_mode")},IF($B$3="Все",$C${rowNumber},$B$3)`,
    `${sr("temporal_mode")},IF($B$4="Все",$D${rowNumber},$B$4)`,
    `${sr("data_scenario")},IF($B$5="Все","*",$B$5)`,
    `${sr("pipeline_tier")},IF($B$6="Все",$E${rowNumber},$B$6)`,
    `${sr("estimand_id")},IF($B$7="Все",$F${rowNumber},$B$7)`,
  ];
  if (summaryIndex.run_family !== undefined) {
    criteria.push(`${sr("run_family")},IF($B$8="Все","*",$B$8)`);
  }
  const common = criteria.join(",");
  const h1Count = `SUMIFS(${sr("n_trials")},${common})`;
  const h0Count = h1Count;
  const effectSum = `SUMIFS(${sr("sum_effect_h1")},${common})`;
  const effectSqSum = `SUMIFS(${sr("sum_effect_sq_h1")},${common})`;
  const seSum = `SUMIFS(${sr("sum_se_h1")},${common})`;
  const covered = `SUMIFS(${sr("coverage_count_h1")},${common})`;
  const fpr = `SUMIFS(${sr("false_positive_count_h0")},${common})`;
  const power = `SUMIFS(${sr("power_count_h1")},${common})`;
  const runtimeSum = `SUMIFS(${sr("runtime_total_ms_h1")},${common})`;
  dashboard.getRange(`G${rowNumber}:N${rowNumber}`).formulas = [[
    `=IFERROR(${h1Count},"")`,
    `=IFERROR(${effectSum}/${h1Count},"")`,
    `=IFERROR(${seSum}/${h1Count},"")`,
    `=IFERROR(SQRT(MAX(0,(${effectSqSum}-(${effectSum}^2/${h1Count}))/(${h1Count}-1))),"")`,
    `=IFERROR(${covered}/${h1Count},"")`,
    `=IFERROR(${fpr}/${h0Count},"")`,
    `=IFERROR(${power}/${h1Count},"")`,
    `=IFERROR(${runtimeSum}/${h1Count},"")`,
  ]];
}
dashboard.tables.add(`A${tableStart}:N${dashboardEnd}`, true, "DashboardSummary");
dashboard.freezePanes.freezeRows(tableStart);
dashboard.getRange(`A${tableStart}:N${dashboardEnd}`).format.font = { name: "Aptos", size: 9 };
dashboard.getRange(`A${tableStart}:N${tableStart}`).format = headerFormat;
dashboard.getRange(`A${tableStart + 1}:N${dashboardEnd}`).format.rowHeight = 18;
dashboard.getRange(`H${tableStart + 1}:J${dashboardEnd}`).format.numberFormat = "0.0000";
dashboard.getRange(`K${tableStart + 1}:M${dashboardEnd}`).format.numberFormat = "0.0%";
dashboard.getRange(`A:A`).format.columnWidth = 35;
dashboard.getRange(`B:B`).format.columnWidth = 44;
dashboard.getRange(`C:F`).format.columnWidth = 21;
dashboard.getRange(`G:N`).format.columnWidth = 15;
dashboard.getRange(`A2:L2`).format.rowHeight = 34;
dashboard.getRange(`A${tableStart}:N${dashboardEnd}`).conditionalFormats.add("colorScale", {
  criteria: [
    { type: "lowestValue", color: "#F4CCCC" },
    { type: "percentile", value: 50, color: "#FFF2CC" },
    { type: "highestValue", color: "#D9EAD3" },
  ],
});

await fs.mkdir(path.dirname(outputPath), { recursive: true });
if (!skipPreviews) {
  await fs.mkdir(previewDir, { recursive: true });
  for (const [sheetName, range, fileName] of [
    ["Dashboard", `A1:N${Math.min(dashboardEnd, 32)}`, "dashboard.png"],
    ["Полные прогоны", `A1:W${Math.min(rawLastRow, 22)}`, "full_runs.png"],
    ["Агрегированные результаты", `A1:${summaryLast}${Math.min(summary.values.length + 1, 24)}`, "aggregates.png"],
    ["Методы", `A1:G${methodRows.length + 1}`, "methods.png"],
    ["Пайплайны", `A1:${excelColumn(catalogHeaders.length - 1)}${Math.min(catalog.values.length + 1, 24)}`, "pipelines.png"],
  ["Источники", `A1:C${sourceRows.length + 1}`, "sources.png"],
  ...(selection.headers.length > 0
    ? [["Выбор лучшего", `A1:${excelColumn(selection.headers.length - 1)}${Math.min(selection.values.length + 1, 24)}`, "selection.png"]]
    : []),
]) {
    const image = await workbook.render({ sheetName, range, scale: 1, format: "png" });
    await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await image.arrayBuffer()));
  }
}
const inspected = await workbook.inspect({
  kind: "table",
  range: `Dashboard!A1:N${Math.min(dashboardEnd, 20)}`,
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 14,
});
console.log(inspected.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
console.log(`saved ${outputPath}`);
process.exit(0);
