import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const payloadPath = path.resolve(process.argv[2] ?? "excel_payload.json");
const outputPath = path.resolve(
  process.argv[3] ?? "outputs/rnd-v3/VarWar_RnD_v3.xlsx",
);
const previewDir = path.resolve(
  process.argv[4] ?? `${outputPath.replace(/\.xlsx$/i, "")}_previews`,
);

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const datasets = Object.fromEntries(
  ["aggregates", "leaders", "pipelines", "methods", "worlds"].map((key) => [
    key,
    Array.isArray(payload[key]) ? payload[key] : [],
  ]),
);

const SHEETS = [
  "Как читать",
  "Дашборд",
  "Лидеры",
  "Диагностика",
  "Все агрегаты",
  "Методы",
  "Пайплайны",
  "Синтетические миры",
];

const LABELS = {
  raw_inference_quality: "Качество до проверки объёма Monte Carlo",
  scientific_sample_size_ok: "Достаточный объём Monte Carlo",
  pipeline: "Пайплайн",
  pipeline_name_ru: "Название пайплайна",
  world_id: "Синтетический мир",
  information_mode: "Режим информации",
  temporal_access: "Временной доступ",
  ranking_estimand_id: "Целевой estimand",
  assignment: "Механизм назначения",
  structure: "Структура рандомизации",
  target_transformer: "Преобразование target",
  analysis_method: "Метод анализа",
  inference_quality: "Качество статвывода",
  inference_quality_score: "Оценка качества статвывода",
  theory_grade: "Уровень теоретического доверия",
  causal_safety: "Причинная безопасность",
  effect_size: "Размер эффекта, σ",
  h0_n_trials: "Число прогонов H0",
  h1_n_trials: "Число прогонов H1",
  h0_rejection_rate: "FPR на H0",
  h0_rejection_ci_low: "FPR: нижняя граница ДИ",
  h0_rejection_ci_high: "FPR: верхняя граница ДИ",
  h1_rejection_rate: "Мощность на H1",
  power: "Мощность",
  type_ii_error: "Ошибка II рода",
  h1_coverage: "Покрытие 95% ДИ",
  h0_coverage: "Покрытие 95% ДИ на H0",
  h1_bias: "Смещение на H1",
  h1_standardized_bias: "Стандартизованное смещение",
  h1_ate_displacement: "Отклонение от ATE",
  h1_rmse: "RMSE относительно estimand",
  h1_ate_rmse: "RMSE относительно ATE",
  h1_empirical_sd: "Эмпирическое SD",
  h1_mean_standard_error: "Средний SE",
  h1_se_to_sd: "Отношение SE / SD",
  h1_mean_ci_length: "Средняя длина ДИ",
  pvalue_superuniform_pass: "Калибровка p-value пройдена",
  fpr_pass: "FPR пройден",
  coverage_pass: "Покрытие пройдено",
  se_sd_pass: "SE / SD пройдено",
  bias_pass: "Смещение пройдено",
  runtime_median_ms: "Медианное время, мс",
  runtime_p95_ms: "Время p95, мс",
  peak_rss_mb: "Пиковая RSS, МБ",
  mde_80: "MDE при мощности 80%",
  variance_reduction_vs_welch: "Снижение дисперсии против Welch",
  status: "Статус",
  method_id: "ID метода",
  category: "Категория",
  summary: "Краткое описание",
  implementation: "Реализация",
  race_scope: "Контур гонки",
};

const FILTERS = [
  ["world_id", "Синтетический мир"],
  ["information_mode", "Режим информации"],
  ["temporal_access", "Временной доступ"],
  ["ranking_estimand_id", "Целевой estimand"],
  ["assignment", "Механизм назначения"],
  ["structure", "Структура рандомизации"],
  ["target_transformer", "Преобразование target"],
  ["analysis_method", "Метод анализа"],
  ["inference_quality", "Качество статвывода"],
];

const DASHBOARD_COLUMNS = [
  "world_id",
  "information_mode",
  "temporal_access",
  "ranking_estimand_id",
  "assignment",
  "structure",
  "target_transformer",
  "analysis_method",
  "inference_quality",
  "scientific_sample_size_ok",
  "theory_grade",
  "causal_safety",
  "effect_size",
  "h0_n_trials",
  "h1_n_trials",
  "h0_rejection_rate",
  "h1_coverage",
  "h1_bias",
  "h1_standardized_bias",
  "h1_ate_displacement",
  "h1_se_to_sd",
  "power",
  "h1_rmse",
  "mde_80",
  "variance_reduction_vs_welch",
  "runtime_median_ms",
  "pipeline",
].filter((key) => datasets.aggregates.some((record) => key in record));

const COLORS = {
  navy: "#17365D",
  blue: "#D9EAF7",
  pale: "#F4F7FA",
  white: "#FFFFFF",
  green: "#C6EFCE",
  greenText: "#006100",
  amber: "#FFEB9C",
  amberText: "#9C6500",
  red: "#FFC7CE",
  redText: "#9C0006",
  deepRed: "#F8696B",
  gray: "#E7E6E6",
  grayText: "#666666",
};

const workbook = Workbook.create();
const sheets = Object.fromEntries(
  SHEETS.map((name) => [name, workbook.worksheets.add(name)]),
);
for (const sheet of Object.values(sheets)) {
  sheet.showGridLines = false;
}

const headerFormat = {
  fill: COLORS.navy,
  font: { name: "Aptos", size: 9, bold: true, color: COLORS.white },
  wrapText: true,
  verticalAlignment: "center",
  borders: { bottom: { style: "medium", color: "#8EA9C1" } },
};
const titleFormat = {
  fill: COLORS.navy,
  font: { name: "Aptos Display", size: 16, bold: true, color: COLORS.white },
  verticalAlignment: "center",
};
const bodyFont = { name: "Aptos", size: 9, color: "#202124" };

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

function normalizeValue(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  if (Array.isArray(value) || (typeof value === "object" && !(value instanceof Date))) {
    return JSON.stringify(value);
  }
  return value;
}

function unionHeaders(rows, preferred = []) {
  const seen = new Set();
  const headers = [];
  for (const key of preferred) {
    if (rows.some((row) => Object.hasOwn(row, key))) {
      headers.push(key);
      seen.add(key);
    }
  }
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        headers.push(key);
        seen.add(key);
      }
    }
  }
  return headers;
}

function displayHeader(key) {
  return LABELS[key] ?? key.replaceAll("_", " ");
}

function writeDataset(sheet, rows, options = {}) {
  const headers = unionHeaders(rows, options.preferred ?? []);
  if (headers.length === 0) {
    sheet.getRange("A1").values = [[options.emptyText ?? "Нет данных для отображения"]];
    sheet.getRange("A1").format = { font: bodyFont, fill: COLORS.amber };
    return { headers: [], lastRow: 1, lastColumn: "A" };
  }
  const values = rows.map((row) => headers.map((key) => normalizeValue(row[key])));
  const lastColumn = excelColumn(headers.length - 1);
  const lastRow = values.length + 1;
  sheet.getRange(`A1:${lastColumn}${lastRow}`).values = [
    headers.map(displayHeader),
    ...values,
  ];
  sheet.getRange(`A1:${lastColumn}1`).format = headerFormat;
  sheet.getRange(`A1:${lastColumn}${lastRow}`).format.font = bodyFont;
  sheet.getRange(`A1:${lastColumn}${lastRow}`).format.verticalAlignment = "center";
  sheet.getRange(`A2:${lastColumn}${lastRow}`).format.rowHeight = 18;
  for (const key of ["summary", "implementation", "description", "purpose_ru"]) {
    const index = headers.indexOf(key);
    if (index < 0) continue;
    const column = excelColumn(index);
    sheet.getRange(`${column}2:${column}${lastRow}`).format.wrapText = true;
    sheet.getRange(`A2:${lastColumn}${lastRow}`).format.rowHeight = 34;
  }
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(options.freezeColumns ?? 1);
  sheet.tables.add(`A1:${lastColumn}${lastRow}`, true, options.tableName);
  applyWidths(sheet, headers);
  applyNumberFormats(sheet, headers, 2, lastRow);
  applyMetricFormatting(sheet, headers, 2, lastRow);
  return { headers, lastRow, lastColumn };
}

function applyWidths(sheet, headers) {
  headers.forEach((key, index) => {
    const column = excelColumn(index);
    let width = 16;
    if (["pipeline", "summary", "implementation", "description", "purpose_ru"].includes(key)) width = 46;
    if (["analysis_method", "target_transformer", "method_id"].includes(key)) width = 30;
    if (["world_id", "assignment", "structure", "ranking_estimand_id"].includes(key)) width = 23;
    if (["information_mode", "temporal_access", "inference_quality", "theory_grade", "causal_safety"].includes(key)) width = 19;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  });
}

function isRate(key) {
  return key.includes("coverage") || key.includes("rejection_rate") || key === "power" ||
    key === "type_ii_error" || key.includes("variance_reduction") || key === "inference_quality_score";
}

function applyNumberFormats(sheet, headers, firstRow, lastRow) {
  if (lastRow < firstRow) return;
  headers.forEach((key, index) => {
    const range = sheet.getRange(`${excelColumn(index)}${firstRow}:${excelColumn(index)}${lastRow}`);
    if (isRate(key)) range.format.numberFormat = "0.0%";
    else if (key.includes("runtime") || key.includes("rss")) range.format.numberFormat = "#,##0.0";
    else if (key.includes("n_trials") || key.endsWith("_count")) range.format.numberFormat = "#,##0";
    else if (
      key.includes("bias") || key.includes("rmse") || key.includes("displacement") ||
      key.includes("effect") || key.includes("se_to_sd") || key.includes("mde")
    ) range.format.numberFormat = "0.0000";
  });
}

function addCellRule(range, operator, formula, fill, fontColor) {
  range.conditionalFormats.add("cellIs", {
    operator,
    formula,
    format: { fill, font: { color: fontColor } },
  });
}

function applyMetricFormatting(sheet, headers, firstRow, lastRow) {
  if (lastRow < firstRow) return;
  const byKey = Object.fromEntries(headers.map((key, index) => [key, excelColumn(index)]));
  const rangesFor = (keys) => keys.filter((key) => byKey[key]).map((key) => ({ key, range: sheet.getRange(`${byKey[key]}${firstRow}:${byKey[key]}${lastRow}`) }));

  for (const { range } of rangesFor(["h0_rejection_rate"])) {
    addCellRule(range, "greaterThan", 0.20, COLORS.deepRed, COLORS.redText);
    addCellRule(range, "between", [0.070000001, 0.20], COLORS.red, COLORS.redText);
    addCellRule(range, "between", [0.03, 0.07], COLORS.green, COLORS.greenText);
    addCellRule(range, "lessThan", 0.03, COLORS.amber, COLORS.amberText);
  }
  for (const { range } of rangesFor(["h1_coverage", "h0_coverage"])) {
    addCellRule(range, "lessThan", 0.90, COLORS.deepRed, COLORS.redText);
    addCellRule(range, "between", [0.90, 0.929999999], COLORS.red, COLORS.redText);
    addCellRule(range, "between", [0.93, 0.99], COLORS.green, COLORS.greenText);
    addCellRule(range, "greaterThan", 0.99, COLORS.amber, COLORS.amberText);
  }
  for (const { range } of rangesFor(["h1_se_to_sd"])) {
    addCellRule(range, "between", [0.90, 1.10], COLORS.green, COLORS.greenText);
    range.conditionalFormats.addCustom(
      `=AND(OR(${range.address.split(":")[0]}<0.9,${range.address.split(":")[0]}>1.1),${range.address.split(":")[0]}>=0.8,${range.address.split(":")[0]}<=1.2)`,
      { fill: COLORS.amber, font: { color: COLORS.amberText } },
    );
    range.conditionalFormats.addCustom(
      `=OR(${range.address.split(":")[0]}<0.8,${range.address.split(":")[0]}>1.2)`,
      { fill: COLORS.red, font: { color: COLORS.redText } },
    );
  }
  for (const { range } of rangesFor(["h1_bias", "h1_standardized_bias", "h1_ate_displacement"])) {
    range.conditionalFormats.add("colorScale", {
      criteria: [
        { type: "lowestValue", color: COLORS.red },
        { type: "num", value: 0, color: COLORS.green },
        { type: "highestValue", color: COLORS.red },
      ],
    });
  }
  for (const { range } of rangesFor(["inference_quality"])) {
    for (const [text, fill, color] of [
      ["good", COLORS.green, COLORS.greenText],
      ["warning", COLORS.amber, COLORS.amberText],
      ["bad", COLORS.red, COLORS.redText],
      ["not_applicable", COLORS.gray, COLORS.grayText],
    ]) {
      range.conditionalFormats.add("containsText", { text, format: { fill, font: { color, bold: true } } });
    }
  }
  for (const { range } of rangesFor(["fpr_pass", "coverage_pass", "bias_pass", "se_sd_pass", "pvalue_superuniform_pass"])) {
    range.conditionalFormats.add("containsText", { text: "true", format: { fill: COLORS.green, font: { color: COLORS.greenText } } });
    range.conditionalFormats.add("containsText", { text: "false", format: { fill: COLORS.red, font: { color: COLORS.redText } } });
  }
  for (const { range } of rangesFor(["causal_safety"])) {
    range.conditionalFormats.add("containsText", { text: "safe", format: { fill: COLORS.green, font: { color: COLORS.greenText } } });
    range.conditionalFormats.add("containsText", { text: "conditional", format: { fill: COLORS.amber, font: { color: COLORS.amberText } } });
    range.conditionalFormats.add("containsText", { text: "post_treatment_risk", format: { fill: COLORS.red, font: { color: COLORS.redText } } });
  }
  for (const { range } of rangesFor(["status"])) {
    range.conditionalFormats.add("containsText", { text: "ran", format: { fill: COLORS.green, font: { color: COLORS.greenText } } });
    range.conditionalFormats.add("containsText", { text: "point_only", format: { fill: COLORS.amber, font: { color: COLORS.amberText } } });
    range.conditionalFormats.add("containsText", { text: "failed", format: { fill: COLORS.red, font: { color: COLORS.redText } } });
  }

  const qualityColumn = byKey.inference_quality;
  for (const key of ["power", "h1_rejection_rate"]) {
    if (!byKey[key] || !qualityColumn) continue;
    const metricColumn = byKey[key];
    const range = sheet.getRange(`${metricColumn}${firstRow}:${metricColumn}${lastRow}`);
    range.conditionalFormats.addCustom(`=AND($${qualityColumn}${firstRow}="good",${metricColumn}${firstRow}>=0.8)`, { fill: COLORS.green, font: { color: COLORS.greenText } });
    range.conditionalFormats.addCustom(`=AND($${qualityColumn}${firstRow}="good",${metricColumn}${firstRow}>=0.5,${metricColumn}${firstRow}<0.8)`, { fill: COLORS.amber, font: { color: COLORS.amberText } });
    range.conditionalFormats.addCustom(`=AND($${qualityColumn}${firstRow}="good",${metricColumn}${firstRow}<0.5)`, { fill: COLORS.red, font: { color: COLORS.redText } });
  }

  for (const { range } of rangesFor(["h1_rmse", "h1_ate_rmse", "mde_80", "runtime_median_ms", "runtime_p95_ms", "peak_rss_mb"])) {
    range.conditionalFormats.add("colorScale", {
      criteria: [
        { type: "lowestValue", color: COLORS.green },
        { type: "percentile", value: 50, color: COLORS.amber },
        { type: "highestValue", color: COLORS.red },
      ],
    });
  }
}

function distinct(key) {
  return [...new Set(datasets.aggregates.map((row) => row[key]).filter((value) => value !== null && value !== undefined && value !== ""))]
    .map(String)
    .sort((a, b) => a.localeCompare(b, "ru"));
}

function writeTitle(sheet, range, text, subtitle) {
  sheet.getRange(range).merge();
  sheet.getRange(range.split(":")[0]).values = [[text]];
  sheet.getRange(range).format = titleFormat;
  sheet.getRange(range).format.rowHeight = 30;
  if (subtitle) {
    const endColumn = range.split(":")[1].replace(/\d+/g, "");
    sheet.getRange(`A2:${endColumn}2`).merge();
    sheet.getRange("A2").values = [[subtitle]];
    sheet.getRange(`A2:${endColumn}2`).format = {
      fill: COLORS.blue,
      font: { name: "Aptos", size: 9, italic: true, color: COLORS.navy },
      wrapText: true,
    };
    sheet.getRange(`A2:${endColumn}2`).format.rowHeight = 32;
  }
}

const guide = sheets["Как читать"];
writeTitle(guide, "A1:F1", "VarWar R&D: как читать результаты", "Рейтинг допустим только внутри одного estimand, мира и информационного контура. Trial-level данные намеренно остаются вне Excel.");
const guideRows = [
  ["Раздел", "Что показывает", "Хороший сигнал", "Плохой сигнал", "Как использовать", "Источник"],
  ["FPR на H0", "Частота ложных срабатываний при истинном нулевом эффекте.", "Около 5%; учитывайте биномиальный ДИ.", "Выше 7% — антиконсервативный вывод; 0 при малом числе прогонов означает неопределённость.", "Жёсткий экран корректности до сравнения мощности.", "https://doi.org/10.2307/2282009"],
  ["Калибровка p-value", "Проверяет super-uniformity p-value под H0 в нескольких порогах.", "Нет систематического избытка малых p-value.", "Избыток малых p-value означает ложную уверенность.", "Смотрите вместе с FPR, а не вместо него.", "https://doi.org/10.1214/18-STS667"],
  ["Покрытие ДИ", "Доля интервалов, содержащих известный GT.", "Около 95%.", "Недопокрытие; покрытие >99% обычно означает чрезмерно широкие интервалы.", "Проверяет качество неопределённости, а не только точечной оценки.", "https://doi.org/10.2307/2331986"],
  ["SE / SD", "Средний заявленный SE к эмпирическому SD оценок.", "0.9–1.1.", "Ниже 0.8 — SE занижен; выше 1.2 — чрезмерно консервативен.", "Диагностика согласованности формулы SE с Монте-Карло.", "https://doi.org/10.1214/12-AOS1008"],
  ["Bias и RMSE", "Смещение и полная ошибка относительно известного estimand.", "Смещение около 0, RMSE ниже внутри сопоставимой группы.", "Значимое смещение или сравнение с чужим estimand.", "Сначала проверяйте estimand, затем ранжируйте.", "https://en.wikipedia.org/wiki/Mean_squared_error"],
  ["Мощность и MDE", "Чувствительность после прохождения валидностного экрана.", "Высокая мощность и низкий MDE.", "Высокая мощность при плохом FPR не считается преимуществом.", "Сравнивайте только строки с quality=good.", "https://doi.org/10.1080/01621459.2012.734175"],
  ["Временной доступ", "pre_treatment_only отделён от полной траектории pre + in-time + post.", "Причинно безопасный pre-only результат.", "full_timeline может содержать leakage и помечается отдельно.", "Не смешивайте эти режимы в одном рейтинге.", "https://mixtape.scunning.com/"],
  ["CUPED / CUPAC", "Снижение дисперсии через заранее доступные ковариаты или прогноз.", "Модель и параметры зафиксированы до outcome либо вывод полностью учитывает обучение.", "Обучение на post-treatment outcome без корректного контракта.", "Смотрите causal_safety и theory_grade.", "https://exp-platform.com/Documents/2013-02-CUPED-Improving%20Sensitivity%20of%20Controlled%20Experiments.pdf"],
  ["Lin adjustment", "Регрессионная коррекция РКИ с взаимодействиями treatment × covariates.", "HC2/HC3 и заранее определённые ковариаты.", "Обычная регрессия без корректного робастного вывода.", "Сильный научный baseline для ковариатной коррекции.", "https://doi.org/10.1214/12-AOS1073"],
  ["Рерандомизация", "Механизм назначения принимает только достаточно сбалансированные сплиты.", "Правило и порог заданы заранее, inference учитывает дизайн.", "Подбор по outcome либо анализ как при полной рандомизации без оговорок.", "Фиксируйте assignment в фильтре.", "https://doi.org/10.1214/12-AOS1008"],
];
guide.getRange(`A4:F${guideRows.length + 3}`).values = guideRows;
guide.getRange("A4:F4").format = headerFormat;
guide.getRange(`A5:F${guideRows.length + 3}`).format = { font: bodyFont, wrapText: true, verticalAlignment: "top" };
guide.getRange(`A5:F${guideRows.length + 3}`).format.rowHeight = 48;
guide.getRange("A:A").format.columnWidth = 24;
guide.getRange("B:E").format.columnWidth = 40;
guide.getRange("F:F").format.columnWidth = 62;
guide.freezePanes.freezeRows(4);

const dashboard = sheets["Дашборд"];
const dashboardLastColumn = excelColumn(Math.max(DASHBOARD_COLUMNS.length, 10));
writeTitle(dashboard, `A1:${dashboardLastColumn}1`, "VarWar R&D: интерактивный дашборд", "Выберите фильтры. Столбец «Показать» вычисляется формулой; затем отфильтруйте его по значению 1. Цвета метрик задаются разными научными правилами.");
dashboard.getRange("A4:B12").values = FILTERS.map(([, label]) => [label, "Все"]);
dashboard.getRange("A4:A12").format = { fill: COLORS.blue, font: { ...bodyFont, bold: true, color: COLORS.navy } };
dashboard.getRange("B4:B12").format = { fill: COLORS.amber, font: { ...bodyFont, bold: true, color: COLORS.navy } };
FILTERS.forEach(([key], index) => {
  dashboard.getRange(`B${index + 4}`).dataValidation = {
    rule: { type: "list", values: ["Все", ...distinct(key)] },
  };
});
dashboard.getRange("D4:H8").values = [
  ["Порядок выбора", "1", "Фиксируйте estimand", "2", "Пройдите валидностный экран"],
  ["Качество вывода", "good", "Научно и численно прошло", "warning", "Нужна осторожность или больше прогонов"],
  ["Запрет", "bad", "Не ранжировать по мощности", "not_applicable", "Только точечная/диагностическая оценка"],
  ["GT-проверки", "FPR", "H0", "coverage/bias", "H0 и H1 относительно известной истины"],
  ["Сравнение", "RMSE", "Меньше лучше", "power", "Больше лучше только при quality=good"],
];
dashboard.getRange("D4:H8").format = { fill: COLORS.pale, font: bodyFont, wrapText: true };
dashboard.getRange("D4:D8").format.font = { ...bodyFont, bold: true, color: COLORS.navy };

const dashboardStart = 15;
const dashboardHeaders = ["Показать", ...DASHBOARD_COLUMNS.map(displayHeader)];
const dashboardLast = excelColumn(dashboardHeaders.length - 1);
const dashboardValues = datasets.aggregates.map((record) => [
  null,
  ...DASHBOARD_COLUMNS.map((key) => normalizeValue(record[key])),
]);
dashboard.getRange(`A${dashboardStart}:${dashboardLast}${dashboardStart + dashboardValues.length}`).values = [
  dashboardHeaders,
  ...dashboardValues,
];
dashboard.getRange(`A${dashboardStart}:${dashboardLast}${dashboardStart}`).format = headerFormat;
dashboard.getRange(`A${dashboardStart + 1}:${dashboardLast}${dashboardStart + dashboardValues.length}`).format.font = bodyFont;
const dashboardIndex = Object.fromEntries(DASHBOARD_COLUMNS.map((key, index) => [key, excelColumn(index + 1)]));
for (let offset = 0; offset < dashboardValues.length; offset += 1) {
  const row = dashboardStart + 1 + offset;
  const clauses = FILTERS.map(([key], filterIndex) => `OR($B$${filterIndex + 4}="Все",$B$${filterIndex + 4}=$${dashboardIndex[key]}${row})`);
  dashboard.getRange(`A${row}`).formulas = [[`=IF(AND(${clauses.join(",")}),1,0)`]];
}
if (dashboardValues.length > 0) {
  const dashboardEnd = dashboardStart + dashboardValues.length;
  dashboard.tables.add(`A${dashboardStart}:${dashboardLast}${dashboardEnd}`, true, "RndV3Dashboard");
  dashboard.getRange(`A${dashboardStart + 1}:A${dashboardEnd}`).format.numberFormat = "0";
  applyNumberFormats(dashboard, ["filter_flag", ...DASHBOARD_COLUMNS], dashboardStart + 1, dashboardEnd);
  applyMetricFormatting(dashboard, ["filter_flag", ...DASHBOARD_COLUMNS], dashboardStart + 1, dashboardEnd);
}
dashboard.freezePanes.freezeRows(dashboardStart);
dashboard.freezePanes.freezeColumns(1);
dashboard.getRange("A:A").format.columnWidth = 11;
applyWidths(dashboard, ["filter_flag", ...DASHBOARD_COLUMNS]);

const leaderInfo = writeDataset(sheets["Лидеры"], datasets.leaders, {
  preferred: DASHBOARD_COLUMNS,
  tableName: "RndV3Leaders",
  emptyText: "Нет лидеров, прошедших валидностный экран.",
  freezeColumns: 2,
});

const diagnosticRows = datasets.aggregates.filter((row) =>
  ["bad", "warning", "not_applicable"].includes(String(row.inference_quality)),
);
const diagnosticInfo = writeDataset(sheets["Диагностика"], diagnosticRows, {
  preferred: [
    ...DASHBOARD_COLUMNS,
    "inference_quality_score",
    "fpr_pass",
    "coverage_pass",
    "bias_pass",
    "se_sd_pass",
    "pvalue_superuniform_pass",
  ],
  tableName: "RndV3Diagnostics",
  emptyText: "Проблемных или неопределённых строк нет.",
  freezeColumns: 2,
});

const aggregateInfo = writeDataset(sheets["Все агрегаты"], datasets.aggregates, {
  preferred: DASHBOARD_COLUMNS,
  tableName: "RndV3Aggregates",
  freezeColumns: 2,
});
const methodInfo = writeDataset(sheets["Методы"], datasets.methods, {
  preferred: ["method_id", "category", "status", "summary", "implementation", "race_scope"],
  tableName: "RndV3Methods",
  freezeColumns: 2,
});
const pipelineInfo = writeDataset(sheets["Пайплайны"], datasets.pipelines, {
  preferred: [
    "pipeline",
    "status",
    "world_id",
    "information_mode",
    "temporal_access",
    "ranking_estimand_id",
    "assignment",
    "structure",
    "target_transformer",
    "analysis_method",
  ],
  tableName: "RndV3Pipelines",
  freezeColumns: 2,
});
const worldInfo = writeDataset(sheets["Синтетические миры"], datasets.worlds, {
  preferred: ["world_id", "name_ru", "family", "purpose_ru", "n_experiment", "r_squared", "treatment_share"],
  tableName: "RndV3Worlds",
  freezeColumns: 2,
});

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

const inspectEnd = Math.min(dashboardStart + dashboardValues.length, dashboardStart + 10);
const inspected = await workbook.inspect({
  kind: "table",
  range: `Дашборд!A1:${dashboardLast}${inspectEnd}`,
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: Math.min(dashboardHeaders.length, 18),
  maxChars: 5000,
});
console.log(`dashboard_inspect=${inspected.ndjson}`);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "rnd v3 formula error scan",
  maxChars: 3000,
});
console.log(`formula_errors=${errors.ndjson}`);

const previewRanges = {
  "Как читать": `A1:F${Math.min(guideRows.length + 3, 30)}`,
  "Дашборд": `A1:${dashboardLast}${Math.min(dashboardStart + dashboardValues.length, 30)}`,
  "Лидеры": `A1:${leaderInfo.lastColumn}${Math.min(leaderInfo.lastRow, 30)}`,
  "Диагностика": `A1:${diagnosticInfo.lastColumn}${Math.min(diagnosticInfo.lastRow, 30)}`,
  "Все агрегаты": `A1:${aggregateInfo.lastColumn}${Math.min(aggregateInfo.lastRow, 30)}`,
  "Методы": `A1:${methodInfo.lastColumn}${Math.min(methodInfo.lastRow, 30)}`,
  "Пайплайны": `A1:${pipelineInfo.lastColumn}${Math.min(pipelineInfo.lastRow, 30)}`,
  "Синтетические миры": `A1:${worldInfo.lastColumn}${Math.min(worldInfo.lastRow, 30)}`,
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const image = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const fileName = `${String(SHEETS.indexOf(sheetName) + 1).padStart(2, "0")}_${sheetName.replaceAll(" ", "_")}.png`;
  await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await image.arrayBuffer()));
}

console.log(JSON.stringify({
  saved: outputPath,
  previews: previewDir,
  sheets: SHEETS.length,
  aggregates: datasets.aggregates.length,
  leaders: datasets.leaders.length,
  diagnostics: diagnosticRows.length,
}));
// artifact-tool keeps native workers alive on Windows; bypass teardown after
// every requested artifact has been flushed successfully.
process.exit(0);
