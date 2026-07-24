# VarWar

VarWar is an R&D workspace for comparing A/B test estimators and variance
reduction wrappers. The old CUPED/CUPAC notebooks are preserved under
`legacy/`; the active project is now a small Python package with synthetic
benchmarks, composable estimators, and agent-facing instructions.

## What Is Implemented

- Canonical statistical baseline: closed-form Welch t-test
  `welch_mean_diff`.
- Integration/parity A/B baseline backed by the PyPI `hypex` package:
  `hypex_ab`.
- HypEx integration baselines: `hypex_cuped` and
  `hypex_cupac_temporal`.
- Local same-dataset adjustment pipelines: `cuped_control__hypex_ab`,
  `cupac_linear_A__hypex_ab`, `cupac_linear_A_plus_B__hypex_ab`, and
  `cupac_linear_A_plus_B_plus_C__hypex_ab`.
- Native estimators/tests: Welch and pooled t-tests, KS test,
  fast approximate KS, energy-distance test, randomized/hedgehog t-test,
  permutation mean test, regression adjustment, ML-rate-style cross-fitted
  adjustment, nearest-neighbor matching, ratio delta method, stratified and
  variance-weighted summaries, robust trimmed/winsorized summaries, and
  Wald/LATE.
- Multiple-testing helpers: Bonferroni, Holm-Bonferroni, and Romano-Wolf
  step-down over caller-supplied bootstrap statistics.
- Design preprocessing: memory-efficient numeric feature binning and genetic
  reduction of atomic category combinations with the A/B-proportional
  variance objective.
- Honest placeholders for methods that need a separate implementation pass:
  anti-nuclear KS, HypEx matcher, calibration/reweighting,
  optimal-transport matching and homogeneity testing, Bayesian CUPAC, and
  prognostic-score block randomization.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
python examples/run_synthetic_race.py
python examples/run_pipeline_race_v2.py --trials 10 --output-dir results/smoke
python examples/run_stratification_race.py
python examples/run_stratification_simulation.py --trials 10
```

Актуальный отчёт полной синтетической гонки находится в
[`results/pipeline_race_v2/report_ru.md`](results/pipeline_race_v2/report_ru.md).
Переработанный Excel-каталог с листами методов, пайплайнов, результатов и
сверки со старой таблицей:
[`outputs/varwar-methods-20260724/VarWar_methods_and_pipeline_race.xlsx`](outputs/varwar-methods-20260724/VarWar_methods_and_pipeline_race.xlsx).
Запуск с `--trials` без `--output-dir` пишет в отдельный каталог
`results/pipeline_race_v2_smoke_<N>` и не перезаписывает канонический отчёт.

## Project Layout

- `src/varwar/` - estimators, target transformers, design preprocessors,
  feature adapters, synthetic data, and benchmark runners.
- `tests/` - lightweight regression tests based on `unittest`.
- `examples/` - user-facing scripts for synthetic races.
- `configs/` - benchmark defaults.
- `docs/methods/` - current Russian method catalog and detailed documentation.
- `docs/` - R&D plan and historical worksheet mappings.
- `legacy/` - previous notebooks and prototype files, kept for context only.

## Design Notes

Estimators and wrappers are separate concepts. A base estimator summarizes an
experiment outcome, while a wrapper transforms the target first and then
delegates to a base estimator. This lets CUPED and CUPAC turn the same HypEx
A/B baseline into new race participants without duplicating inference code.

Design preprocessing is evaluated in a separate race because it changes the
experiment design rather than directly estimating an effect. See
`docs/methods/README.md` for the taxonomy and `docs/methods/all_methods.md` for
the complete method/status registry.

The current `hypex_ab` implementation is a native compatibility adapter for the
HypEx A/B baseline slot. It first tries the PyPI `hypex` package, then falls
back to the local Welch-compatible implementation when HypEx is unavailable in
the runtime. Canonical pipeline-race definitions set `require_pypi=True`, so a
failed HypEx integration becomes an explicit failed row instead of a silent
fallback. The adapter can also be wired to a concrete callable:

```bash
export VARWAR_HYPEX_BACKEND="my_project.hypex_backend:estimate"
```

The backend callable should accept `(data, spec)` and return `EstimateResult`.
Where a mature library function is both clear and efficient, VarWar uses it,
for example SciPy for t/KS p-values when SciPy is installed.

The local CUPAC chain uses the following data-zone contract:

```text
A = temporal CUPAC on pre-treatment lag2 -> lag1 data
B = a separate control-fitted linear adjustment on expert-safe in-time features
C = a separate control-fitted linear adjustment after safety and balance gates
```

The chain is trained from the one available experiment dataset: current
outcome is never used to train stage A; stages B and C are fitted only on the
current control. `hypex_cupac_temporal` remains an integration/parity row, not
a predecessor in the local chain:

```text
hypex_ab
  -> cupac_linear_A__hypex_ab
  -> cupac_linear_A_plus_B__hypex_ab
  -> cupac_linear_A_plus_B_plus_C__hypex_ab
```
