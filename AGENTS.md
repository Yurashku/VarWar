# Agent Instructions

## Mission

Build VarWar as a reliable R&D harness for sensitivity and variance reduction
research in A/B testing. Keep the code clean enough that estimators can be
compared on synthetic data now and on internal real data later.

## Ground Rules

- Do not copy implementation code from the HSE repository. Use it only as a
  research map for method names and experiment ideas.
- Keep old VarWar artifacts in `legacy/` unless the user explicitly asks to
  revive one.
- Prefer vectorized NumPy/Pandas operations and closed-form estimators before
  simulation-heavy approaches.
- Use mature library implementations where they are well-defined and efficient
  for the task, such as SciPy tests. Do not add a heavy dependency merely to
  avoid a small closed-form implementation.
- Avoid hidden global state. Estimators should be deterministic for fixed data
  and fixed seeds.
- If an estimator is not understood well enough, add a documented placeholder
  instead of inventing a fake implementation.

## Architecture

- Base estimators implement `estimate(data, spec) -> EstimateResult`.
- Target wrappers implement `fit_transform(data, spec, auxiliary_data=None)`.
- `WrappedEstimator` composes a target wrapper with any base estimator.
- The default race must include `hypex_ab`, `cuped+hypex_ab`, `hypex_cupac`,
  `sklearn_cupac_A+hypex_ab`, `sklearn_cupac_A_plus_B+hypex_ab`, and
  `sklearn_cupac_A_plus_B_plus_C+hypex_ab`.
- Use the PyPI `hypex` package for HypEx baselines. If an internal wrapper is
  needed, wire it through `VARWAR_HYPEX_BACKEND="module:function"` or
  `HypExABTestEstimator(backend=...)`.

## Validation

Run at least:

```bash
python -m unittest discover -s tests
python examples/run_synthetic_race.py
```

When adding a method, add a synthetic scenario that can fail if the estimator is
biased, unstable, or omitted from the race registry.
