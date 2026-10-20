# Дизайн-документ: ревизия VarWar R&D v3 и план пересчёта

**Репозиторий:** `/home/ivan/projects/VarWar`, `main @ ab7e49a`. Документ сводит подтверждённые находки шести линз ревью: worlds, estimators, targets, runner, results, design.

---

## 0. Очистка не выполнена: вы просили её, но эта задача её запрещала

Вы просили почистить локально мусор от старого R&D. Ни одна стадия этого ревью ничего не удалила: задача workflow прямо запрещала менять файлы в репозитории. Ниже точный список, чтобы чистку можно было сделать одним следующим шагом после вашего «да».

| Что | Размер | Где лежит | Можно ли удалять |
|---|---|---|---|
| `~/.cache/pip` | **382 МБ** | вне репо, не R&D | безопасно (`pip cache purge`), но нужно ваше «да» |
| `.pytest_cache` | 40 КБ | в репо, не в git (untracked) | безопасно |
| scratch-скрипты ревью (`/tmp/claude-1000/.../scratchpad/v3review/`) | ≈13+ МБ | вне репо | безопасно |
| `results/pipeline_race_v2` | 3.9 МБ | в git (tracked) | нужен commit; на него ссылаются `configs/pipeline_race_v2.toml:36`, `docs/methods/pipelines.md:58`, `AGENTS.md:67`, `docs/rnd_plan.md:54` |
| `results/rct_pipeline_race_cartesian_smoke` | 4.2 МБ (в т.ч. `raw_results.csv` 2.6 МБ рядом с его `.gz`-копией) | в git (tracked) | нужен commit; минимум — удалить несжатый `.csv` |
| `results/rct_pipeline_race_final` | 7.9 МБ | в git (tracked) | нужен commit |
| `outputs/rct-pipeline-race`, `outputs/varwar-methods-20260724` | 1.0 МБ + 92 КБ | в git (tracked) | нужен commit |
| 2 одинаковых `pipeline_matrix.parquet` (md5 `a4e255ea…`), 4 файла `excel_payload.json` | 22 + 11.6 МБ | в git (tracked) | восстанавливаются из кода; можно добавить в `.gitignore` |
| `results/rnd_v3/{smoke,screening}` | 2.3 + 3.1 МБ | в git (tracked) | устарели (несвязанные seed); будут пересчитаны (§D) |

**Не трогать:**
- `Эстиматоры.xlsx` — ваш собственный файл, не в git.
- `.venv` (614 МБ) и `src/varwar.egg-info` — нужны для editable-установки.
- `legacy/` — `AGENTS.md` велит держать старые артефакты там, пока вы явно не попросите иное.
- `run_pipeline_race_v2.py` и `run_stratification_race.py` — `AGENTS.md`, раздел Validation, требует их запускать, так что это не мусор. Удалять их можно только вместе с правкой `AGENTS.md`.

**Сколько реально освободится.** Удаление tracked-файлов освобождает в рабочей копии всего ≈30–40 МБ. Их байты остаются в истории git (pack 38 МБ). Основной выигрыш — кэш pip (382 МБ).

---

## A. Вердикт по v3

### A.1. Что надёжно (по убыванию уверенности)

1. **Вычисление агрегатов.** FPR, bias, RMSE, SD, mean SE, coverage, power и VR по `trial_results.parquet` пересчитываются в `aggregates.parquet` до 4-го знака. Все 288 000 + 60 000 + 16 800 строк имеют статус `ran`. Формула Уилсона, поправка Holm и truth-based метрики корректны (`validation.py:45-191`). Повторный запуск бит-в-бит детерминирован.
2. **Арифметика финалистов.**
   - `RegressionAdjustmentEstimator` (Lin): $\hat\beta_T$ и SE в вариантах HC2/HC3 совпадают со statsmodels до $10^{-15}$.
   - Без ковариат HC2 совпадает с Welch.
   - Welch и pooled Student совпадают со scipy.
   - Monte Carlo-покрытие Lin HC2, ML-RATE и stratified DiM ≈ 0.95, в том числе при гетероскедастичности, гетерогенном эффекте и разбиении 30/70. Единственное исключение: при $n=80$ покрытие HC2 0.942 ± 0.005.
3. **Дизайн эксперимента внутри мира.**
   - Назначение не видит исходы: `_OUTCOME_COLUMNS` защищён guard-проверкой.
   - Независимость эмпирически подтверждена: z дисбаланса по $y_0$ имеет var 1.007, доля $|z|>1.96$ равна 0.052.
   - CRN корректен: H0 и H1 используют одно и то же назначение, все пайплайны одного механизма видят общее назначение.
   - Утечек outcome/treatment в A/B/C-обучение не найдено: A обучается как lag2→lag1, B и C обучаются на контроле с cross-fit.
4. **Упорядочение по RMSE и variance reduction.** Эти метрики не зависят от SE. Порядок Lin < ML-RATE < CUPAC A < CUPED hist < Welch надёжен в мире confirmation, а по RMSE Lin лучше CUPAC A во всех 24 мирах.

### A.2. Что ненадёжно (по убыванию влияния на выводы)

| # | Проблема | Что ломает | Severity |
|---|---|---|---|
| 1 | **SE для преобразованного target считается по сырому $y$.** `AnalysisMethod.analyze` (`analysis.py:312-318`) передаёт в `inference.infer` исходные frame и spec. `Welch`, `PooledStudent` и `DesignBasedNormal` (`inference.py:83,111,146`) вызывают `_groups(data, spec)` на `y`. `StudentizedRandomization` и `BootstrapT` (177-190, 283-297) стьюдентизируют на тот же сырой SE. | Все FPR, coverage, SE/SD, p-value, power и MDE для CUPED/CUPAC во всех стадиях. Подтверждение: SE(cuped) = SE(cupac) = SE(DIM) точно (std 0) во всех 57 600 парных строках. В confirmation mean SE = 0.141324 у трёх пайплайнов. | critical |
| 2 | **В мире population_shift «H0» не нулевая.** `effect=0` обнуляет только константную часть эффекта; модификатор $0.35\,z(x_1+0.5x_3)$ стандартизован по популяции, а когорта смещена к высоким $x_1$. В итоге $\overline{\mathrm{SATE}}_{H0}=0.2848$. | Все 5 вердиктов `bad` в этом мире. «FPR = 1.000» на деле является мощностью. Если считать относительно истины, размер теста 0.015–0.055. | major |
| 3 | **Вердикты при R=200 отражают MC-шум.** Порог bias $\lvert b\rvert/SD\le0.10$ — точечный жёсткий отказ: несмещённая оценка проваливает его с $P=0.157$. Гейт `good` требует FPR ≤ 0.030 и coverage ≥ 0.970 ($P=0.124$ каждое). Super-uniformity — жёсткий отказ без поправки на множественность по строкам. | Из 28 уникальных `bad` 14 вызваны только bias с CI ∋ 0: heteroskedastic, interactions, temporal_b/c у Lin и ML. Лидеры подбираются по «удачно консервативной» реализации. В `rnd_v3/screening` лидеров 0. | major |
| 4 | **Доказательства дублируются.** Четыре контекста бит-идентичны (72 000/72 000 групп). Seed переиспользуются между шардами: 24 мира получают всего 6 потоков случайности. Пары `strong_prior_mismatch ≡ gaussian_r2_04_n2000` и `panel_shift ≡ temporal_drift` совпадают для identity-пайплайнов. | Межмировые подсчёты завышены: 960 строк агрегатов содержат 120 уникальных вердиктов. Реально независимых DGP заметно меньше 24. | major |
| 5 | **DGP по построению проигрывает CUPED/CUPAC.** `pre_target` = $f(x_1,x_2)$ + iid-шум, без персистентного unit-эффекта и без $x_3$. Остаток `pre_target_lag1` после $(x_1,x_2)$ — чистый шум. | Вывод «Lin/ML-RATE лучше CUPED/CUPAC» в значительной мере зашит в DGP. Миры panel_shift, drift и mismatch не проверяют то, что обещают их имена. | major |
| 6 | **Охват гонки узкий.** В гонке были только полная рандомизация и 5 вручную отобранных пайплайнов. ML-RATE — линейный ridge на тех же признаках, что и Lin. Нет мира, где неравное разбиение сочетается с гетерогенностью. Мощность насыщена в 20 из 21 мира с $n=2000$. | Нельзя сказать «лучшая полная процедура» и нельзя отделить Lin от ML-RATE. | major |
| 7 | **Результаты нельзя воспроизвести.** `rnd_v3/{smoke,screening,confirmation}` посчитаны до `ab7e49a`, с несвязанными seed. `confirmation_pipeline_ids()` на screening возвращает все 9 пайплайнов, а финалистов 5 — их выбрали вручную. | Цепочка screening → finalists → confirmation не аудируется. | major |
| 8 | **Отчёт.** `rnd_report.py:84-88` сортирует вердикт как строку, поэтому `bad` идёт первым. RSS не заполняется, а там, где измерен, это RSS всего процесса (~1.4 ГБ у всех). `mde_80` = NaN во всех 40 строках confirmation. Ранжирование смешивает effect sizes. | Ложное впечатление, что «Lin/ML-RATE везде bad». | major / minor |
| 9 | **Стратифицированная рандомизация смещена.** Детерминированный tie-break остатков (`assignment.py:348-356`) даёт неравные $P(T{=}1\mid s)$, упорядоченные по прогнозу. | Сейчас не влияет: эти дизайны не участвовали в гонке. Но при включении Welch получит bias −0.039 (z = −6.4). | major (латентный) |

### A.3. Какие утверждения README и report_ru.md выдерживают проверку

| Утверждение | Статус |
|---|---|
| «Текущий confirmation подтверждает только мир `gaussian_r2_04_n200`» | **Держится.** Добавить: «и только при полной рандомизации». |
| «confirmation — независимые 1000 реализаций пяти финалистов» | **Частично.** 1000 реализаций и новый seed — да. Но финалисты выбраны вручную, а inference для CUPED/CUPAC неверен. |
| «Screening требует ≥200 общих реализаций» | **Нет** для `rnd_v3/screening`: там 3 seed на (world, trial), реализации не общие. Для all-worlds — да. |
| Lin HC2 / ML-RATE / Welch валидны в `gaussian_r2_04_n200` | **Держится:** FPR 0.048 / 0.047 / 0.047, coverage 0.95. |
| CUPED hist `good`, CUPAC A `warning` (confirmation) | **Не держится.** Метрики основаны на сыром SE. |
| all-worlds `report_ru.md`: Lin/ML-RATE `bad` в population_shift и heteroskedastic | **Не держится:** это артефакты из A.2 #2, #3 и #8. |
| «Проверка ... время и RSS-память», правило выбора по MDE и RSS | **Не держится:** RSS не различает пайплайны, MDE = NaN. |
| «Несопоставимые estimand никогда не ранжируются вместе»; A+B и A+B+C помечены `conditional`/`post_treatment_risk` | **Держится.** |
| «Интеграционные бейзлайны `hypex==1.0.6`» | **Устарело:** установлена 1.0.7, где исправлен баг $\theta=\rho$. Методы HypEx запускались только в smoke. |

---

## B. Наиболее перспективный пайплайн

**Канонический ID:**
```
adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2
```
**По-человечески:** полная рандомизация, без адаптера признаков и без преобразования target, регрессионная корректировка Лина (центрированные ковариаты, взаимодействия с treatment, OLS) с робастной ковариацией HC2 и t-калибровкой.

### B.1. Числа

**Confirmation** (`results/rnd_v3/confirmation`, `gaussian_r2_04_n200`, $R=1000$, seed 20260807):

| Метрика | Lin HC2 | ML-RATE | Welch | MC SE (≈) |
|---|---|---|---|---|
| FPR (H0) | 0.048 [0.036, 0.063] | 0.047 | 0.047 | 0.0068 |
| Coverage, δ = 0.1 / 0.2 | 0.949 / 0.954 | 0.948 / 0.955 | 0.953 / 0.955 | 0.0069 |
| Bias, δ = 0.2 | −0.0044 | −0.0024 | −0.0035 | 0.0034 |
| SD / mean SE, δ = 0.2 | 0.1077 / 0.1109 | 0.1092 / 0.1122 | 0.1354 / 0.1410 | — |
| SE/SD | 1.02–1.03 | 1.02–1.03 | 1.00–1.04 | ≈0.022 ($1/\sqrt{2R}$) |
| RMSE, δ = 0.2 | 0.1078 | 0.1092 | 0.1354 | — |
| VR относительно Welch | 0.41 / 0.37 | 0.39 / 0.35 | 0 | — |
| MSE/MSE$_{\text{Welch}}$ | 0.633 ± 0.040 | — | 1 | — |
| Power при δ = 0.2 | 0.404 | 0.405 | 0.265 | 0.0155 |
| Медиана runtime | 6.6 мс | 16.5 мс | 5.3 мс | — |

Парная разность квадратов ошибок Lin − ML-RATE: z = −3.7, то есть у Lin MSE ≈ на 2.5% ниже. Верификатор это число не пересчитывал.

**Сверка с теорией.**
- Ожидаемая VR ≈ $R^2\cdot$(поправка на $2K/n$): $1-0.6\cdot(1+10/200)=0.37$. Наблюдается 0.367.
- Ожидаемая мощность Lin при δ = 0.2 и bias −0.0044: $1-\Phi\big((t_{0.975,188}\cdot0.1109-0.1956)/0.1077\big)\approx0.415$. Наблюдается 0.404.
- То же для Welch: ≈0.273, наблюдается 0.265.

**All-worlds screening** ($R=200$, 23 мира, population_shift исключён):
- FPR ≤ 0.075, coverage ≥ 0.935 во всех мирах.
- MSE/MSE$_{\text{Welch}}$ от 0.195 (`r2_08_n2000`) до ≈0.98 (high-dim, zero-inflated). Хуже Welch только в `gaussian_r2_00_n200`: 1.062 ± 0.023.
- По RMSE Lin первый в 7 из 9 миров, где вообще есть `good`-лидеры. В остальных двух первыми оказались Welch (r2_00_n200) и ML-RATE (lognormal).
- Если пересчитать population_shift относительно истинного SATE, размер Lin = 0.050.

**Теоретическое обоснование** (Lin 2013; Li & Ding 2017).
- При полной рандомизации интерактивная OLS асимптотически не хуже ни DIM, ни ANCOVA без взаимодействий — при любом $p$ и любой гетерогенности.
- HC2 состоятельна для консервативной Neyman-дисперсии.
- Без ковариат HC2 совпадает с Welch, поэтому этот метод — естественное обобщение канонического бейзлайна VarWar.

### B.2. Насколько сильны доказательства

- **Сильные** для одного мира ($n=200$, гауссов, $R^2=0.4$, $R=1000$): MC SE FPR ≈ 0.007, SE/SD ≈ 0.022.
- **Умеренные** для остальных 23 миров. $R=200$ даёт MC SE FPR ≈ 0.015. Миры коррелированы через seed, часть из них — копии друг друга. Effect sizes связаны через CRN, поэтому δ = 0.1 и δ = 0.2 не являются независимыми подтверждениями.
- **Нет** доказательств для дизайнов кроме полной рандомизации, для PATE при сдвиге популяции, для сценария «неравное разбиение + гетерогенность», для малых $n$ с большим $K$.

### B.3. Где Lin HC2 может проиграть

1. **Малое $n$ при большом $K$ и низком $R^2$.** У Lin $2K+2$ параметров (здесь 12) и $df=n-12$. В `r2_00_n200` MSE +6% к Welch. При $n=80$ и разбиении 30/70 HC2 недопокрывает: 0.942 ± 0.005.
2. **Нелинейный прогноз.** В quadratic, threshold и interactions линейный $R^2$ ≪ целевого (0.21 / 0.29 / 0.06 против 0.4). Нелинейный ML-RATE (GBM) потенциально выиграет, но в гонке такого learner нет.
3. **Тяжёлые хвосты и выбросы** (student_t3, contaminated, lognormal). OLS неробастна, а робастные треки (Yuen, Catoni, MoM) в гонке не участвовали.
4. **Корректный CUPED/CUPAC в DGP, где pre-period метрика — продолжение $y$.** Тогда pooled-$\theta$ CUPAC ≈ ANCOVA по прогнозу, и разница с Lin может исчезнуть. Сейчас это не проверено.
5. **Rerandomization и стратификация.** Lin почти ничего не добавляет к уже сбалансированному дизайну: RMSE 0.106–0.113 против 0.1095 при полной рандомизации. При этом SE HC2 станет консервативной.
6. **PATE при гетерогенности.** HC2 не учитывает член $\gamma^\top\Sigma_X\gamma/n$, поэтому для PATE она антиконсервативна (B.4, пункт е).

### B.4. Как это реализовано в VarWar

| Где | Что |
|---|---|
| Реестр | `src/varwar/analysis_registry.py` — `_native_method("lin_regression__hc2", RegressionAdjustmentEstimator(covariance_type="HC2"), estimand_id="sample_ate", grade=A, safety=SAFE, primary=True)` (≈ стр. 199–206) |
| Inference | native: `NativeEstimatorInference` (`analysis.py:373-389`) берёт SE из `native_payload`. Поэтому баг A.2 #1 Lin **не затрагивает**. |
| Точка + SE | `src/varwar/estimators.py:RegressionAdjustmentEstimator.estimate` (221–282) |
| Ковариация | `estimators.py:_robust_covariance` (56–73): `pinv(X'X)`, $h_{ii}$ через `einsum`, $\omega_i=e_i^2/\max(1-h_{ii},10^{-12})$ |
| p / CI | `estimators.py:_build_result` (76–102): `t_two_sided_p(stat, df)`, `t_critical(alpha, df)`, $df=n-p$ |
| Ковариаты в race | `rnd_runner.py:_spec` (190–202): `pre_treatment_cols=("pre_target_lag1","x1","x2","x3","x4")`, итого $K=5$, $p=12$ |

**Спецификация.** Пусть $T_i\in\{0,1\}$, а $\tilde X_i = X_i-\bar X$ — ковариаты, центрированные по выборке. Модель:

$$
Y_i=\alpha+\tau T_i+\tilde X_i^\top\beta+T_i\tilde X_i^\top\gamma+e_i,\qquad \hat\tau_{\text{Lin}}=\hat\tau .
$$

Закрытая форма через OLS по каждой руке отдельно:

$$
\hat\tau_{\text{Lin}}=\big[\bar Y_1-(\bar X_1-\bar X)^\top\hat\beta_1\big]-\big[\bar Y_0-(\bar X_0-\bar X)^\top\hat\beta_0\big].
$$

HC2-дисперсия:

$$
\hat V_{\text{HC2}}=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\operatorname{diag}\!\Big(\tfrac{\hat e_i^2}{1-h_{ii}}\Big)\mathbf X(\mathbf X^\top\mathbf X)^{-1},\qquad \widehat{SE}=\sqrt{[\hat V]_{TT}}.
$$

CI: $\hat\tau\pm t_{1-\alpha/2,\,n-p}\,\widehat{SE}$.

**Расхождения реализации с теорией и с документацией:**
- (а) Центрирование делается через `np.nanmean` по **всем** строкам до маски `valid` (стр. 246, затем 251–253). При пропусках $\hat\tau$ перестаёт быть SATE анализируемых строк. В синтетике пропусков нет, поэтому эффект нулевой.
- (б) Калибровка $t_{n-p}$ вместо нормальной (у Lin асимптотика нормальная). Это слегка консервативно и разумно, но при малом $n$ корректнее Bell–McCaffrey $df$.
- (в) Ковариаты только числовые: `region` и `device` игнорируются. Это дизайнерское решение.
- (г) В текущем DGP `pre_target_lag1` — шумовой регрессор при уже известных $x_1, x_2$. Он стоит одну степень свободы и два параметра.
- (д) HC2 для SATE консервативна на величину $S^2_{\tau\mid X}/n$. В гомогенных мирах эта добавка равна нулю, отсюда SE/SD ≈ 1.02.
- (е) Поправки на PATE в методе **нет**. Трек `population_ate` требует отдельной реализации (F.3).
- (ж) Вырожденные плечи не обрабатываются явно: `lstsq` и `pinv` скрывают коллинеарность. Стоит выводить ранг в diagnostics.

---

## C. План исправлений кода

Правки минимально-инвазивные, в духе `AGENTS.md`: компоненты остаются раздельными, публичные ID не меняются, новые ID проходят через `method_catalog` и `docs/methods/all_methods.md`, к каждому методу прилагается сценарий, на котором он может упасть.

### P0 — без этого пересчёт бессмыслен

| # | Файл | Суть правки | Тест, который это ловит |
|---|---|---|---|
| P0-1 | `pipeline_execution.py` (`TransformingPointEvaluator`, `DirectPointEvaluator`), `inference.py` (`WelchInference`, `PooledStudentInference`, `DesignBasedNormalInference`, `StudentizedRandomizationInference`, `BootstrapTInference`) | Добавить `evaluator.materialize(data, spec, context) -> (frame, spec)`. В `Direct` это identity; в `Transforming` — рабочий frame с колонкой `__varwar_target_stage_k` и `current_spec`. Каждый non-native inference вызывает `_groups` на материализованных frame/spec, в том числе **внутри каждой** перестановки или бутстреп-реплики после refit. Убрать `del evaluator`. | `tests/test_inference_transformed_target.py`: (1) `cuped_historical`+Welch: SE совпадает с Welch по adjusted-колонке до 1e-12 и отличается от SE identity при $\theta\ne0$; (2) studentized randomization: знаменатель реплики равен SE adjusted; (3) MC при $n=400$, 200 повторов, $\rho\approx0.9$: SE/SD ∈ [0.85, 1.15] (сейчас 2.6). |
| P0-2 | `validation.py:summarize_trial_group` (135–137), `rnd_runner.py:637,699` | Размер теста считать как отклонение $H_0:\theta=\theta_{\text{true}}$, то есть через CI-инверсию относительно truth. FPR «против 0» оставить только при $\max_r\lvert\theta_r\rvert<10^{-9}$. Иначе ставить `h0_is_null=False` и не применять к строке FPR, super-uniformity и sharp-null тесты. | `tests/test_validation.py::test_nonnull_h0_flagged`: синтетический агрегат с truth = 0.285 даёт `h0_is_null=False` и вердикт по размеру относительно truth. Регрессия на population_shift: size ≤ 0.07 у Lin. |
| P0-3 | `validation.py:202-243` | (а) Bias: `hard_bad` только если $\lvert\hat b\rvert-1.96\,\mathrm{mcse}>0.10\cdot\widehat{SD}$; если превышен лишь точечный порог — `uncertain`. (б) FPR и coverage: `bad`, если Wilson-lower FPR > 0.07 или Wilson-upper coverage < 0.93; `conservative` — это флаг, а не условие для `good`. (в) SE/SD: `bad` только при CI < 0.90; > 1.10 — `conservative`. (г) Super-uniformity: BH по строкам стадии, $q=0.05$. | `tests/test_validation_mc.py`: 2000 синтетических агрегатов несмещённой калиброванной оценки при $R=200$ дают долю `bad` < 2% (сейчас ≈16%). Смещение 0.3 SD при $R=200$ даёт `bad` > 90%. FPR 0.12 при $R=1000$ даёт `bad`. |
| P0-4 | `rnd_runner.py:589-603` | Seed = `SeedSequence([config.seed, zlib.crc32(world_id), trial]).spawn(2)` — отдельные потоки для мира и для назначения. Результат больше не зависит от состава и порядка шардов. | `tests/test_rnd_runner.py::test_seed_independent_of_world_order`: прогон `[A,B]` и `[B]` даёт для B бит-идентичные результаты; у двух разных миров seed различаются. |
| P0-5 | `examples/merge_rnd_shards.py` | Проверять уникальность ключа (world, trial, pipeline, context, effect), одинаковость конфигурации шардов (seed, R, effects, версия кода и git sha) и сохранять полный config в `run_metadata.json`. | `tests/test_merge_shards.py`: дублирующиеся ключи или несовпадающий seed дают ошибку. |
| P0-6 | `rnd_report.py:84-88` | Упорядоченный категориальный порядок `good < warning < bad`, удаление mode-invariant дублей. Для `bad` выводить причину (bias / FPR / coverage / p-value). | `tests/test_rnd_report.py`: при наличии `good` первая строка таблицы — `good`; дубли контекстов схлопнуты. |

### P1 — корректность гонки и экономия расчёта

| # | Файл | Суть | Тест |
|---|---|---|---|
| P1-1 | `rnd_runner.py:302-321, 648-717` | Флаг `consumes_panel_prior` / `consumes_in_time` из метаданных компонентов. Mode-invariant пайплайн исполняется один раз и транслируется на остальные контексты с пометкой `mode_invariant=True`. Ожидаемая экономия ≈75% расчёта. | Результаты транслированного и честно исполненного контекста совпадают; число исполнений уменьшается в 4 раза. |
| P1-2 | `synthetic_v3.py:234-237` | Опция `effect_modifier_centering ∈ {population, cohort}`. Для sample_ate-трека population_shift — `cohort`, чтобы получилась настоящая sharp/weak null. population_ate-трек остаётся на `population`. | `generate_world(population_shift, effect=0, centering="cohort")` даёт $\lvert\mathrm{SATE}\rvert<10^{-12}$. |
| P1-3 | `assignment.py:348-356` | Tie-break остатков через `assignment_rng`: перестановка до `argmax` или систематическое округление. | По 2000 seed частота лечения в каждой страте равна share ± 3 MC SE. Prognostic + Welch: $\lvert z_{\text{bias}}\rvert<3$. |
| P1-4 | `rnd_runner.py:922-934, 884-913` | В группу ранжирования добавить `effect_size`. Финалистов выбирать **только** через `confirmation_pipeline_ids` и сохранять список вместе с правилом в metadata. | Лидеры не смешивают δ. В metadata screening есть `finalists` и `selection_rule`. |
| P1-5 | `rnd_runner.py:838-858` | try/except на уровне одного effect, а не всей группы. Колонки `failure_rate`, `nan_pvalue_rate`; `failure_rate > 0` даёт warning. Аналитическая MDE: $(t_{0.975}+t_{0.8})\cdot\overline{SE}$ с MC CI. | Искусственный сбой в одном effect не выбрасывает H0 и другие effect. `mde_80` не NaN. |
| P1-6 | `target_registry.py`, `transforms.py:286-304` | Добавить в core race `cupac_hypex_compatible_A` (pooled $\theta$) и зарегистрированный CUPED с $\theta=\widehat{\mathrm{Cov}}/\widehat{\mathrm{Var}}$ по эксперименту (`CupedTransformer(fit_data="experiment_all")`, сейчас не зарегистрирован). Диагностика $\mathrm{Var}(Y^*)/\mathrm{Var}(Y)$. | В `strong_prior_mismatch` отношение дисперсий у pooled-$\theta$ ≤ 1.05, а у CUPAC A с фиксированным коэффициентом > 1.5 (регрессия на текущее поведение). |
| P1-7 | `estimators.py:246`, `estimators.py:131-134`, `robust_estimators.py:62-73` | Центрировать по `valid`. Перемешивать `train_idx` RNG фолда перед inner-split. Писать долю упавших бутстреп-реплик в diagnostics; > 5% считать отказом inference. | Lin с NaN в части $X$ совпадает со statsmodels на `valid`-строках. ML-RATE на отсортированном и перемешанном frame даёт одинаковое распределение. |
| P1-8 | `target_registry.py:390-393`, `README.md:23`, `estimators.py:1942,1990` | Обновить версию до `hypex==1.0.7`, явно передать `equal_variance=False`. | Тест: строка версии совпадает с `importlib.metadata.version("hypex")`. |

### P2 — дизайн экспериментов и научный охват

| # | Где | Суть |
|---|---|---|
| P2-1 | `synthetic_v3.py:225-239, 343-369` | Единый процесс: $Y_{it}=\alpha_i+f(X_i)+d_t+\varepsilon_{it}$, где $y$ — период $T{+}1$ того же ряда с персистентным $\alpha_i$. Panel shift и mismatch вносить только во **внешнюю** панель (без сущностей когорты). Логировать oracle $\mathrm{corr}^2(y,\text{pre})$ и oracle $R^2$ мира. Мир «связь с историей поменялась» держать отдельно. |
| P2-2 | `synthetic_v3.py` (каталог) | Новые миры: $p\in\{0.2,0.3\}$ с $\tau(x)$, коррелированным с прогностическим $x$, и с арм-специфичной дисперсией; ярус $n=500$; `noise_*` в `covariate_cols` для high-dim; прогностические редкие страты; temporal_b/c_unsafe со сдвигом среднего, переживающим стадию A. |
| P2-3 | `estimators.py`, `method_catalog.py`, `docs/methods/all_methods.md` | Новый ID `ml_rate__crossfit_gbm_hc2` (GBM или RF с cross-fit, вместе с region/device) и сценарий, где он обязан обогнать Lin (quadratic, threshold). |
| P2-4 | `analysis_registry.py`, `inference.py` | Inference с учётом rerandomization: асимптотика Li–Ding–Rubin $\sqrt{1-R^2}\,\varepsilon+\sqrt{R^2}\,L_{K,a}$ или рандомизационный тест с replay правила принятия (`replay_trace` уже хранит порог). Отдельная design-гонка. Добавить `pre_target_lag1` в критерии баланса. |
| P2-5 | `rnd_runner.py:98-157`, `profile_rnd_v3_memory.py` | Память на вызов: пик `tracemalloc` внутри estimate/infer или RSS за вычетом baseline после импорта. Кэшировать матрицу на 762k строк по хэшу реестра. |
| P2-6 | `docs/methods/statistical_validation.md` | Синхронизировать с кодом: MDE, coverage под H0, SE-scale, Pareto, p ≤ α. |
| P2-7 | `estimators.py:_build_result` | Опциональные Bell–McCaffrey $df$ для HC2 при $n<300$. |

---

## D. Дизайн пересчёта

### D.1. Порядок

0. **Перед пересчётом:** P0-1…P0-6 и P1-1, P1-2, P1-4, P1-5, P1-6. P1-3 и P2-1/P2-2 делать до стадии S1 только если вы согласны менять DGP (вопрос G-3).
1. **Зафиксировать заранее** (§D.4): правила, R, seed, список миров и пайплайнов. Положить их в `configs/rnd_v3_1_plan.toml` (описательный файл) и закоммитить **до** первого запуска.
2. **Стадии:**

| Стадия | Миры | Пайплайны | Effects | R | Base seed |
|---|---|---|---|---|---|
| S0 smoke (локально) | `gaussian_r2_04_n200`, `contaminated` | smoke-набор (28) | 0, 0.2 | 5 | 20261001 |
| S0b mini-check (локально) | `gaussian_r2_04_n200`, `population_shift`, `strong_prior_mismatch`, `heteroskedastic` | Welch, Lin HC2, ML-RATE, CUPED hist, CUPAC A, CUPAC pooled-$\theta$ | 0, 0.2 | 60 | 20261002 |
| S1 screening (дома) | все миры каталога (24, плюс новые, если есть P2-2) | Welch; Lin HC2/HC3; ML-RATE ridge (и GBM при наличии P2-3); CUPED hist; CUPED pooled-$\theta$; CUPAC A; CUPAC pooled-$\theta$ A; A+B и A+B+C для temporal_*; stratified (atomic, genetic) для many_rare_strata; population_weighted / PATE-трек для population_shift | 0, 0.1, 0.2 | 500 | 20261010 |
| C1 confirmation (дома) | все миры | финалисты S1, выбранные **кодом** (≤ 6), плюс Welch как якорь | 0, 0.1, 0.2 | 2000 | 20261020 (непересекающийся) |
| P1 power (дома) | 8 репрезентативных миров (по одному на класс) | 2–3 лидера C1 плюс Welch | 0, 0.05, 0.1, 0.15, 0.2, 0.3 | 1000 | 20261030 |
| D1 design race (опционально, после P1-3/P2-4) | core-миры при $n=200$ и $n=2000$ | финалисты × {CR, rerand Mahalanobis, rerand mixed, prognostic blocks, genetic strata} | 0, 0.2 | 500 | 20261040 |

Почему $R=500$ для screening: MC SE bias/SD ≈ 0.045, FPR ≈ 0.0097. Почему $R=2000$ для confirmation: MC SE FPR ≈ 0.0049, так что FPR 0.065 отличим от 0.05 примерно на 3σ; bias/SD ≈ 0.022.

### D.2. Шардирование и merge

После P0-4 seed зависят от `world_id`, поэтому шардировать можно по мирам произвольно:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for w in $(python -c "from varwar.synthetic_v3 import WORLD_CATALOG as W; print(' '.join(W))"); do
  python examples/run_rnd_v3.py --stage screening --trials 500 --seed 20261010 \
    --world "$w" $(cat configs/rnd_v3_1_screening_pipelines.args) \
    --output-dir "results/rnd_v3_1/shards/$w" &
  # ограничить параллелизм по RAM: ~1.4 ГБ на процесс до кэширования матрицы (P2-5)
done; wait
python examples/merge_rnd_shards.py results/rnd_v3_1/shards/* --output results/rnd_v3_1/screening
```

Merge обязан падать при дублях ключей и при расхождении конфигурации (P0-5). Финалисты для C1 выводятся `confirmation_pipeline_ids(aggregates)` и записываются в metadata (P1-4).

### D.3. Время и память

Опорные замеры (2 потока):
- генерация мира и пересборка реестра: 0.17–0.38 с на (world, effect, trial);
- медианы пайплайнов на trial: Welch 3.5 мс, CUPED 4.1, Lin 6.7, CUPAC A 10.5, ML-RATE 17;
- построение матрицы ≈12 с на запуск, RSS ≈ 1.4 ГБ на процесс;
- текущий all-worlds core: 0.67 CPU-ч пайплайнов + ≈1 CPU-ч генерации.

| Стадия | Единиц (world × effect × trial) | Оценка CPU-ч | Wall при 8 ядрах | Пиковая RAM |
|---|---|---|---|---|
| S0 smoke | 20 | ≈0.04 (около 2 мин wall) | ≈2 мин | 1.4 ГБ |
| S0b | 4 × 2 × 60 = 480 | ≈0.05 + 4 × 12 с матрицы | ≈3–4 мин в одном процессе | 1.4 ГБ |
| S1 | ≈28 × 3 × 500 = 42 000 | ≈0.3 с (мир) + ≈0.1 с (≈12 аналитических пайплайнов) → ≈4.7; с GBM ML-RATE +≈3.5 (оценка, GBM не замерен) | ≈1–1.5 ч | 8 × 1.4 ≈ 11 ГБ (при 16 ГБ брать ≤ 6 шардов) |
| C1 | ≈28 × 3 × 2000 = 168 000 | ≈19 (+≈14 с GBM) | ≈3–4 ч | как S1 |
| P1 | 8 × 6 × 1000 = 48 000 | ≈5 | ≈40 мин | как S1 |
| D1 | ≈4 × 2 × 500 × 5 дизайнов = 20 000 | ≈3 (rerand ≈9 попыток назначения) | ≈30 мин | как S1 |

Итого ≈30–50 CPU-ч, то есть одна ночь на 8 ядрах. Без P1-1 (дедупликация контекстов) часть на пайплайны умножается на 4. Resampling-inference в полной матрице (≈10 000 CPU-ч) и AutoML × resampling (≈480 000 CPU-ч) в этот план **не входят**: сначала пруннинг по итогам S1.

### D.4. Критерии приёмки (зафиксировать до запуска)

**Корректность кода** (S0, S0b; детерминированные проверки):
- 133 существующих теста плюс новые тесты P0 — зелёные.
- SE(`cuped_historical`+Welch) равна Welch по adjusted-колонке; SE(cuped) ≠ SE(identity).
- population_shift при `centering=cohort` даёт $\lvert\mathrm{SATE}_{H0}\rvert<10^{-12}$; без этой опции строка помечается `h0_is_null=False`.
- Повторный запуск бит-идентичен; результат мира не зависит от состава шарда.
- В отчёте нет дублей контекстов, первая строка таблицы — не `bad`.

**Валидность процедуры** (C1, на каждой паре мир × процедура):
- Размер: Wilson-upper FPR ≤ 0.07 (при $R=2000$ это наблюдаемая FPR ≲ 0.060), относительно truth.
- Coverage: Wilson-lower ≥ 0.93.
- Bias: $\lvert\hat b\rvert+1.96\,\mathrm{mcse}\le0.10\cdot\widehat{SD}$.
- SE/SD: нижняя граница CI ≥ 0.90; значение > 1.10 — флаг `conservative`, не провал.
- Super-uniformity: не отвергнута после BH ($q=0.05$) по всей стадии.

**Выбор лучшего пайплайна** (среди валидных во **всех** мирах своего класса):
- Основная метрика — средняя по классам миров $\log(\mathrm{MSE}/\mathrm{MSE}_{\text{Welch,CR}})$. Классы задаются по DGP после схлопывания дубликатов, мир-копия не учитывается дважды.
- Победитель обгоняет второе место по парной MSE (CRN) с 95% CI, исключающим 0, в большинстве классов и ни в одном классе не невалиден.
- Иначе — ничья, а tie-break по медиане runtime.
- Мощность и MDE80 берутся из P1 только как описательные метрики.

**Гипотеза, проверяемая заранее.** «Lin HC2 входит в топ-2 по MSE во всех классах линейных миров, а GBM-ML-RATE обгоняет его в нелинейных». Её подтверждение или опровержение фиксируется в отчёте.

### D.5. Локально или дома

- **Локально (≤ 5 мин на шаг):** `python -m unittest discover -s tests` (~95 с + новые тесты), S0 smoke, S0b mini-check.
- **Дома:** S1, C1, P1, D1 с шардированием по мирам. Каждый шард — отдельный процесс с `OMP_NUM_THREADS=1`; число шардов ограничивать по RAM. Результаты коммитить только после merge и проверок.

---

## E. Legacy: что ещё актуально, что исправлено

| Замечание legacy-ревью | Статус в v3 |
|---|---|
| Pooled Student t как основной тест | **Исправлено.** Канон — Welch; pooled и sharp-null помечены `diagnostic_only`. Под weak null при неравных дисперсиях они действительно невалидны (FPR ≈ 0.19–0.21), Welch и studentized permutation — нет. |
| Нерандомизированный DGP с IV | **Исправлено.** Назначение не зависит от исходов, независимость подтверждена эмпирически. |
| In-sample CUPAC | **Исправлено частично.** B и C обучаются на контроле с cross-fit, A — временная (lag2→lag1). **Новая проблема:** у A фиксированный коэффициент 1, а у historical CUPED $\theta$ — AR-наклон pre-периода; дисперсия раздувается до 1.8–2.2× (P1-6). |
| Баг HypEx CUPED $\theta=\rho$ (≤ 1.0.5) | **Исправлено** в 1.0.7 (`Cov/Var`; у CUPAC — $\mathrm{Cov}(\hat y,y)/\mathrm{Var}(\hat y)$). Строки версий в VarWar устарели (1.0.6); HypEx как эталон в гонках не участвовал (P1-8). |
| Общий паттерн legacy: обёртка меняет точку, а дисперсию считает по исходным данным | **Повторился в v3** в новой архитектуре: inference на сыром target (A.2 #1, P0-1). |

---

## F. План PDF-отчёта по выбранному пайплайну

**Название:** «Регрессионная корректировка Лина с HC2 для A/B-тестов: теория, реализация в VarWar и численная проверка».

### F.1. Разделы
1. Постановка задачи: потенциальные исходы, SATE и PATE, полная рандомизация, ковариаты до воздействия.
2. Эстиматор: модель, закрытая форма, связь с DIM, ANCOVA и CUPED.
3. Асимптотика и дисперсия: Neyman-консервативность, HC2 и HC3, $t$-калибровка, малые $n$.
4. PATE-поправка и сдвиг популяции.
5. Обобщение через cross-fit (ML-RATE) и условия, при которых оно выигрывает.
6. Мощность и MDE.
7. Реализация в VarWar: файл:функция, отличия от теории (B.4).
8. Численная проверка: стенд, миры, правила валидности (D.4), результаты.
9. Где метод проигрывает; рекомендации к применению.
10. Ограничения и что было исправлено (ссылки на §C).

### F.2. Ключевые формулы

*Эстиматор и закрытая форма:*
$$
\hat\tau_{\text{Lin}}=\big[\bar Y_1-(\bar X_1-\bar X)^\top\hat\beta_1\big]-\big[\bar Y_0-(\bar X_0-\bar X)^\top\hat\beta_0\big].
$$
Без ковариат это $\bar Y_1-\bar Y_0$. При $\hat\beta_1=\hat\beta_0=\hat\theta$ и одной ковариате получается ANCOVA/CUPED.

*Асимптотика* (Lin 2013; Li & Ding 2017). Пусть $e_i(a)$ — остаток проекции $Y_i(a)$ на $(1,X_i)$. Тогда
$$
\sqrt n(\hat\tau-\tau_{\text{SATE}})\xrightarrow{d}\mathcal N\!\Big(0,\ \tfrac{S^2_{e(1)}}{p}+\tfrac{S^2_{e(0)}}{1-p}-S^2_{e(1)-e(0)}\Big),
$$
причём дисперсия Lin ≤ дисперсии DIM и ANCOVA при любом $p$.

*HC2 и HC3:*
$$
\hat V_{\text{HC}k}=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\operatorname{diag}\!\Big(\tfrac{\hat e_i^2}{(1-h_{ii})^{m}}\Big)\mathbf X(\mathbf X^\top\mathbf X)^{-1},\quad m=1\ (\text{HC2}),\ m=2\ (\text{HC3}).
$$
Тождество HC2 = Welch без ковариат. Консервативность для SATE: $\mathbb E\hat V_{\text{HC2}}-V_{\text{SATE}}\approx S^2_{e(1)-e(0)}/n\ge0$.

*PATE-поправка.* Пусть $\hat\gamma=\hat\beta_1-\hat\beta_0$, $\mu^*$ — средние ковариат в целевой популяции. Тогда
$$
\hat\tau_{\text{PATE}}=\hat\tau+(\mu^*-\bar X)^\top\hat\gamma,\qquad
\widehat{\mathrm{Var}}=c^\top\hat V_{\text{HC2}}\,c+\hat\gamma^\top\hat\Sigma_{\mu^*}\hat\gamma,\quad c=e_T+(\mu^*-\bar X)\otimes e_{T\times X}.
$$
При $\mu^*=\bar X$ и iid-выборке из популяции получается $\hat V_{\text{HC2}}+\hat\gamma^\top\hat\Sigma_X\hat\gamma/n$. В VarWar этого пока нет.

*Cross-fit (ML-RATE).* $\hat g^{(-k)}$ обучается вне фолда $k$, затем
$$
Y_i=\alpha+\tau T_i+\beta\,(\hat g_i-\bar{\hat g})+\delta\,T_i(\hat g_i-\bar{\hat g})+e_i,
$$
с HC2 и $df=n-4$. Валидность не зависит от качества $\hat g$, потому что $T\perp X$; эффективность определяется $\mathrm{corr}^2(Y,\hat g)$.

*Мощность и MDE:*
$$
\mathrm{MDE}_{1-\beta}=(t_{1-\alpha/2,df}+t_{1-\beta,df})\,SE,\qquad SE\approx\sigma_Y\sqrt{1-R^2}\sqrt{\tfrac1{n_1}+\tfrac1{n_0}}\cdot\big(1+\tfrac{K}{n}\big).
$$
Пример: $n=200$, $R^2=0.4$ дают MDE80 ≈ 2.82 · 0.111 ≈ 0.31σ против 0.40σ у Welch. Поэтому сетка {0.1, 0.2} и не доходила до 80% мощности.

### F.3. Таблицы и рисунки

| Элемент | Брать сейчас из v3 | Заменить после пересчёта |
|---|---|---|
| Табл. 1: confirmation Lin, ML-RATE, Welch (FPR с Wilson, coverage, bias ± MC SE, SD, SE, SE/SD, RMSE, VR, power) | `results/rnd_v3/confirmation/aggregates.parquet`, только 3 identity-строки, один контекст; оговорить: старый runner, несвязанные seed | на C1 ($R=2000$, все миры) |
| Строки CUPED и CUPAC | **не брать** (неверный SE) | после P0-1 |
| Табл. 2: MSE/MSE$_{\text{Welch}}$ и размер относительно truth по мирам | all-worlds screening, без population_shift и без дублей, с пометкой «R=200, коррелированные seed» | C1 по классам миров |
| Рис. 1: ECDF p-value Lin под H0 с полосой | confirmation trial_results | C1 |
| Рис. 2: forest plot coverage и FPR по мирам с Wilson CI | all-worlds (иллюстративно) | C1 |
| Рис. 3: VR против oracle $R^2$ мира с линией $1-R^2(1+2K/n)$ | частично (миры n2000) | S1/C1 с логированием oracle $R^2$ (P2-1) |
| Рис. 4: кривая мощности и MDE80 | нет | P1 |
| Рис. 5: SE/SD Lin HC2 и HC3 при малом $n$ и разбиении 30/70 | MC из ревью (n = 80/200), с оговоркой | отдельный прогон |
| Табл. 3: runtime | медианы из trial_results (Lin 6.6 мс, ML 16.5 мс) | C1 |
| RSS | не брать | после P2-5 |

---

## G. Открытые вопросы к вам

1. **Очистка:** разрешаете ли (а) `pip cache purge` (382 МБ), (б) удаление `.pytest_cache` и scratch-файлов ревью, (в) commit с удалением tracked-каталогов старых гонок, `outputs/rct-pipeline-race` и дублирующих parquet/json вместе с правкой ссылок в `AGENTS.md`, `configs` и `docs`? Нужно ли при этом убрать из `AGENTS.md` команды `run_pipeline_race_v2.py` и `run_stratification_race.py`, или они остаются частью валидации?
2. **Приоритет estimand:** что основное — SATE или PATE? От этого зависит, реализовывать ли PATE-поправку Lin в P1.
3. **Менять ли DGP (P2-1, P2-2) до пересчёта?** Это ломает сравнимость с v3, но без этого сравнение CUPED/CUPAC с Lin остаётся предрешённым.
4. **Реальные данные:** типичные $n$, доли разбиения, типы метрик (ratio, count, тяжёлые хвосты), есть ли кластеризация (пользователь → сессии), есть ли pre-period значение той же метрики? Это определяет приоритет миров и методов (delta-method, кластерные SE, робастные треки).
5. **Бюджет на домашней машине:** сколько ядер и RAM? От этого зависят число шардов и обязательность P2-5 (кэш матрицы).
6. **Нужны ли 4 контекста** (`experiment_only` / `panel_prior` × `pre_treatment_only` / `full_timeline`) в основном отчёте, или достаточно показывать их только для пайплайнов, которые их реально потребляют?
7. **Design race (D1)** входит в текущий объём или откладывается?
8. **Нелинейный ML-RATE (GBM):** добавлять сейчас? Это +≈50% времени C1.
9. **HC2 или HC3 по умолчанию** для малых $n$, и нужны ли Bell–McCaffrey $df$?
10. **HypEx 1.0.7 как внешний эталон** в S1 (CUPED/CUPAC parity) — включать?
11. **PDF:** чем собирать (LaTeX или pandoc), нужен ли английский абстракт, кто аудитория (внутренняя команда или внешний читатель)?

---

**Использованные пути:**
- Код: `/home/ivan/projects/VarWar/src/varwar/{estimators.py,inference.py,analysis.py,pipeline_execution.py,validation.py,rnd_runner.py,rnd_report.py,synthetic_v3.py,assignment.py,target_registry.py,transforms.py,analysis_registry.py}`, `/home/ivan/projects/VarWar/examples/{run_rnd_v3.py,merge_rnd_shards.py}`.
- Результаты: `/home/ivan/projects/VarWar/results/rnd_v3/confirmation/aggregates.parquet`, `/home/ivan/projects/VarWar/results/rnd_v3_all_worlds_core/screening/`.
- Scratch-файлы линз: `/tmp/claude-1000/-home-ivan-projects-VarWar/a1b714b4-f849-4bc6-bea6-fe7b2a4f4103/scratchpad/v3review/`.