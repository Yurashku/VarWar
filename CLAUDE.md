# Claude Notes

This repository is now an estimator R&D harness, not a notebook dump.

Important defaults:

- The active source lives in `src/varwar`.
- Old notebooks and prototypes live in `legacy`.
- HypEx baselines use the PyPI `hypex` package when available and are
  represented by `hypex_ab`, `cuped+hypex_ab`, and `hypex_cupac`.
- CUPAC is not one layer: keep the RnD6 A/B/C feature-chain variants separate.
- HSE code should not be imported or translated line by line.

When working autonomously:

1. Inspect `docs/rnd_plan.md` and `docs/estimator_catalog.md`.
2. Inspect `docs/estimator_sheet_v02.md` before changing race membership.
3. Add or update tests before claiming an estimator is ready.
4. Keep placeholder methods explicit and searchable.
5. Preserve memory/runtime discipline. Do not introduce bootstrap, CV, or
   model-heavy defaults unless the benchmark config opts into them.
