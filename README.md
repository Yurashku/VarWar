# VarWar

VarWar is an R&D workspace for comparing A/B test estimators and variance
reduction wrappers. The old CUPED/CUPAC notebooks are preserved under
`legacy/`; the active project is now a small Python package with synthetic
benchmarks, composable estimators, and agent-facing instructions.

## What Is Implemented

- Baseline A/B estimator backed by the PyPI `hypex` package when available:
  `hypex_ab`.
- Wrapper estimators: `cuped+hypex_ab`, `hypex_cupac`, and the local CUPAC
  RnD6 chain.
- CUPAC feature-set variants from RnD reports: `sklearn_cupac_A+hypex_ab`,
  `sklearn_cupac_A_plus_B+hypex_ab`, and
  `sklearn_cupac_A_plus_B_plus_C+hypex_ab`.
- Native estimators/tests: Welch and pooled t-tests, KS test,
  fast approximate KS, energy-distance test, randomized/hedgehog t-test,
  permutation mean test, regression adjustment, ML-rate-style cross-fitted
  adjustment, nearest-neighbor matching, ratio delta method, stratified and
  variance-weighted summaries, robust trimmed/winsorized summaries, and
  Wald/LATE.
- Multiple-testing helpers: Bonferroni, Holm-Bonferroni, and Romano-Wolf
  step-down over caller-supplied bootstrap statistics.
- Honest placeholders for methods that need a separate implementation pass:
  anti-nuclear KS, HypEx matcher, calibration/reweighting, HSE genetic
  algorithm, optimal-transport matching, and Bayesian CUPAC.

The HSE repository is used only as a list of research directions. Its code is
not copied into this project.

## Quick Start

```powershell
$py = "C:\Users\ivan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py -m unittest discover -s tests
& $py examples\run_synthetic_race.py
```

For a regular Python environment:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
python examples/run_synthetic_race.py
```

## Project Layout

- `src/varwar/` - estimator interfaces, wrappers, synthetic data, and benchmark
  runner.
- `tests/` - lightweight regression tests based on `unittest`.
- `examples/` - user-facing scripts for synthetic races.
- `configs/` - benchmark defaults.
- `docs/` - R&D plan and estimator catalog.
- `legacy/` - previous notebooks and prototype files, kept for context only.

## Design Notes

Estimators and wrappers are separate concepts. A base estimator summarizes an
experiment outcome, while a wrapper transforms the target first and then
delegates to a base estimator. This lets CUPED and CUPAC turn the same HypEx
A/B baseline into new race participants without duplicating inference code.

The current `hypex_ab` implementation is a native compatibility adapter for the
HypEx A/B baseline slot. It first tries the PyPI `hypex` package, then falls
back to the local Welch-compatible implementation when HypEx is unavailable in
the runtime. It can also be wired to a concrete callable:

```bash
export VARWAR_HYPEX_BACKEND="my_project.hypex_backend:estimate"
```

The backend callable should accept `(data, spec)` and return `EstimateResult`.
Where a mature library function is both clear and efficient, VarWar uses it,
for example SciPy for t/KS p-values when SciPy is installed.

The CUPAC chain follows `Yurashku/rnd_reports` RnD6:

```text
A = pre-treatment features
B = expert-safe in-time features
C = balance-gated in-time features
```

`hypex_cupac` is kept as a reference/parity slot. The main predecessor chain is
`hypex_ab -> sklearn_cupac_A -> sklearn_cupac_A_plus_B -> sklearn_cupac_A_plus_B_plus_C`.
