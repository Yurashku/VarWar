# Композиция методов

Рекомендуемый порядок:

```text
historical numeric columns
    ↓ NumericFeatureBinner.fit/transform
historical categorical features
    ↓ GeneticStrataReducer.fit
frozen category → stratum mapping
    ↓ transform(experiment)
experiment with stratum
    ↓ optional CUPED/CUPAC target transformation
stratified estimator
    ↓ optional multiple-testing correction
final inference
```

## Допустимые композиции

- `genetic_strata_reducer → stratified_mean_diff`
- `numeric_feature_binner → genetic_strata_reducer → stratified_mean_diff`
- `genetic_strata_reducer → CUPED → stratified_mean_diff`
- `numeric_feature_binner → genetic_strata_reducer → CUPAC → stratified_mean_diff`

## Недопустимые композиции

- Обучать binner boundaries на экспериментальном outcome.
- Обучать GA mapping на post-treatment target.
- Подбирать `n_strata` по p-value текущего эксперимента.
- Считать GA самостоятельным effect estimator.

Для одиночного сравнения используется `run_stratification_race`. Для серии
симуляций результаты объединяются и передаются в
`summarize_stratification_trials`, который рассчитывает bias, RMSE, empirical
standard deviation и rejection rate (FPR при true effect 0, power иначе).
Готовый null/alternative прогон находится в
`examples/run_stratification_simulation.py`; число повторов задаётся через
`--trials` или `configs/stratification_simulation.toml`.
