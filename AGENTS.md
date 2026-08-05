# Agent Instructions

## Mission

Build VarWar as a reliable R&D harness for sensitivity and variance reduction
research in A/B testing. Keep the code clean enough that estimators can be
compared on synthetic data now and on internal real data later.

## Ground Rules

- Treat external research repositories as reference material. Implement from
  a verified mathematical specification and document only VarWar behavior.
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
- Keep assignment mechanisms, randomization-structure builders, feature
  adapters, target transformers and registered `AnalysisMethod` pairs distinct.
- Keep every public method ID and maturity status in `varwar.method_catalog`.
  Update `docs/methods/all_methods.md` in the same change.
- Use category-specific non-executable placeholders. Never register a target
  transformer or design preprocessor as an estimator merely to expose its name.

## Architecture

- The canonical race unit is `AnalysisMethod = PointEstimator +
  InferenceProcedure + CompatibilityContract`; do not form a free Cartesian
  product of estimators and inference procedures.
- Target wrappers implement `fit_transform(data, spec, auxiliary_data=None)`.
  Primary race pipelines must work from one available experiment dataset;
  `auxiliary_data` is an explicit opt-in compatibility path, never an implicit
  default.
- `WrappedEstimator` is legacy-only. New resampling must refit the complete
  target chain through `TransformingPointEvaluator`.
- `GeneticStrataReducer` is a `RandomizationStructureBuilder`; benchmark it
  either in the dedicated stratification race or as the structure component of
  a complete v3 procedure, never as a standalone estimator.
- `NumericFeatureBinner` is a target-free `FeatureAdapter`. Fit its boundaries
  on reference data and reuse them unchanged on experiment data.
- The default race must include `welch`, `hypex_ab`, `hypex_cuped`,
  `cuped_control__hypex_ab`, `hypex_cupac_temporal`,
  `cupac_linear_A__hypex_ab`, `cupac_linear_A_plus_B__hypex_ab`, and
  `cupac_linear_A_plus_B_plus_C__hypex_ab`.
- CUPAC A is temporal `lag2 -> lag1` prediction. B and C are separate
  control-fitted linear stages; they must not be folded into the A model.
- Use `welch_mean_diff` as the canonical no-adjustment statistical baseline;
  keep HypEx rows as integration/parity baselines.
- Use the PyPI `hypex` package for HypEx baselines. If an internal wrapper is
  needed, wire it through `VARWAR_HYPEX_BACKEND="module:function"` or
  `HypExABTestEstimator(backend=...)`.

## Validation

Run at least:

```bash
python -m unittest discover -s tests
python examples/run_synthetic_race.py
python examples/run_stratification_race.py
python examples/run_pipeline_race_v2.py --trials 5 --output-dir results/smoke/pipeline_race_v2
python examples/run_rnd_v3.py --stage smoke
```

When adding a method, add a synthetic scenario that can fail if the estimator is
biased, unstable, or omitted from the race registry.
