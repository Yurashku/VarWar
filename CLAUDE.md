# Claude Notes

This repository is now an estimator R&D harness, not a notebook dump.

Important defaults:

- The active source lives in `src/varwar`.
- Old notebooks and prototypes live in `legacy`.
- HypEx baselines use the PyPI `hypex` package when available and are
  represented by `hypex_ab`, `hypex_cuped`, and `hypex_cupac_temporal`.
- `welch_mean_diff` is the canonical statistical baseline. HypEx rows measure
  package parity and wrapper integration.
- CUPAC is not one layer: keep A as temporal `lag2 -> lag1` CUPAC and B/C as
  separate control-fitted linear stages with distinct admission policies.
- External implementations are research references only. Public documentation
  must describe the algorithms and guarantees implemented in VarWar.
- Genetic strata reduction is a design preprocessor with a separate race;
  numeric binning is a target-free feature adapter.

When working autonomously:

1. Inspect `docs/methods/README.md` and the relevant detailed method page.
2. Inspect `docs/estimator_sheet_v02.md` before changing race membership.
3. Add or update tests before claiming an estimator is ready.
4. Keep placeholder methods explicit and searchable.
5. Preserve memory/runtime discipline. Do not introduce bootstrap, CV, or
   model-heavy defaults unless the benchmark config opts into them.
6. Keep `src/varwar/method_catalog.py` and `docs/methods/all_methods.md` in sync.
