# Передача R&D VarWar v3.1 домашнему агенту

Документ — полный контекст для агента, который продолжает R&D на домашней машине.
Читать целиком перед любыми действиями. Обязательные правила репозитория — в
`AGENTS.md` и `CLAUDE.md`, они имеют приоритет.

---

## 0. Безопасность и гигиена репозитория (сначала это)

1. **Работать только в свежем клоне.** 20–21.10.2026 история `Yurashku/VarWar`
   была переписана: из неё удалены выводы ноутбуков с персональными данными
   клиентов. Затем репозиторий удалён и создан заново под тем же именем. Любой клон,
   сделанный раньше, содержит утёкшие коммиты: его нельзя `pull`/`push`, его нужно
   удалить и склонировать заново.
   ```bash
   git clone https://github.com/Yurashku/VarWar.git && cd VarWar && git checkout rerun/v3.1
   ```
2. **Никаких реальных данных в git.** Не коммитить строки, `df.head()`, выводы
   ноутбуков с данными, идентификаторы клиентов, внутренние имена полей. Ноутбуки в
   `legacy/` хранятся без выводов: так и оставить.
3. **Не переписывать историю и не делать force-push** без явной просьбы пользователя.
4. Работа идёт в ветке `rerun/v3.1`. Результаты коммитить туда же (см. §6),
   в `main` — только по просьбе пользователя.

---

## 1. Цель R&D

Найти **лучшую полную процедуру подведения итогов рандомизированного A/B-теста**:
связку «дизайн → назначение → преобразование метрики → оценка + статистический вывод».
Критерии в порядке приоритета:

1. **Валидность вывода:** размер теста, покрытие CI, калибровка SE, отсутствие смещения
   относительно правильного estimand.
2. **Эффективность среди валидных:** MSE/RMSE, снижение дисперсии, мощность, MDE.
3. Время и память — как tie-break.

Несопоставимые estimand (`sample_ate`, `population_ate`, `robust_native`) и режимы
доступа к данным (`experiment_only`/`panel_prior` × `pre_treatment_only`/`full_timeline`)
никогда не ранжируются вместе.

---

## 2. Что уже сделано (история вопроса)

| Этап | Итог |
|---|---|
| Legacy R&D (`legacy/`, 2025) | Невалиден: pooled Student t, нерандомизированный IV-DGP, in-sample CUPAC; цифры 34%/33% на реальных данных — in-sample. Только исторический контекст. |
| R&D v3 (`main`, авг. 2026) | Архитектура хорошая, но найдено 46 подтверждённых проблем: `docs/review/v3_review_design.md`. |
| **v3.1 (`rerun/v3.1`, 20.10.2026)** | Исправлены P0 и P1 из ревью и часть P2; 205 тестов зелёные; smoke и mini-check пройдены; полный пересчёт **не запускался** — это ваша задача. |

### Ключевые исправления v3.1 (чтобы не «чинить» их обратно)

- **Inference на преобразованной метрике.** Ранее Welch / Student / normal /
  permutation / bootstrap-t для CUPED/CUPAC считали SE по сырому `y`: SE/SD было
  2.5, после исправления 1.0. Механизм: `evaluator.materialize(...)`, kwarg
  `materialized=` у `infer` (`analysis.py`, `pipeline_execution.py`, `inference.py`).
- **Настоящая H0 по estimand.** Модификатор эффекта центрируется по когорте для
  `sample_ate` и по популяции для `population_ate` (`spec_for_estimand`). Размер теста
  меряется относительно истинного estimand. Старое «FPR = 1.0 в population_shift» было
  артефактом разметки.
- **Вердикты с учётом MC-ошибки** (`validation.py`, описание в
  `docs/methods/statistical_validation.md`): границы Уилсона, bias с mcse,
  BH для super-uniformity, вердикты `good < warning < uncertain < bad`,
  флаг `conservative` не считается провалом.
- **Seed** = `SeedSequence([seed, crc32(world_id), trial])`; результат не зависит
  от состава шардов. Merge шардов проверяет ключи, конфиг и git SHA.
- **Mode-invariant broadcast:** пайплайн, не читающий panel prior и in-time
  колонки, исполняется один раз на четыре контекста; бит-идентичность проверена.
- **Финалисты выбираются кодом** (`select_confirmation_finalists`), включая
  исключение пайплайнов с полным отказом хотя бы в одном мире.
- **Новые миры** (всего 37):
  - панельная семья — исход продолжает тот же ряд, что и pre-period: `*_panel`;
  - кейс Фридмана — `freedman_p20/p30_panel`;
  - `nonlinear_panel`, ярус `n500`, high-dim.
- **Новые методы:**
  - `ml_rate__crossfit_gbm_hc2`;
  - `lin_regression__hc2_pate` (только `population_ate`);
  - targets `cuped_pooled_theta` и `cupac_linear_pooled_theta_A`;
  - опция Bell–McCaffrey df у Лина (в гонке не зарегистрирована).
- hypex обновлён до 1.0.7: в нём исправлен баг θ=ρ в CUPED, `equal_variance=False`
  передаётся явно.

---

## 3. Текущий лидер (гипотеза, которую нужно подтвердить или опровергнуть)

```
adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2
```

Регрессия Лина: OLS с центрированными pre-treatment ковариатами и их взаимодействиями
с воздействием, HC2-сэндвич, квантиль $t_{n-p}$. Теория, реализация и численная проверка
изложены в отчёте `outputs/reports/VarWar_Lin_HC2_report_ru.pdf` (исходники —
`docs/reports/lin_hc2/`). **Эмпирика отчёта пока основана на результатах v3
до исправлений.** Строки Лина, ML-RATE и Welch там корректны, потому что используют
нативную SE.

Что известно на сейчас:

- **Подтверждение v3** (`gaussian_r2_04_n200`, R = 1000): FPR 0.048, покрытие 0.95,
  MSE на 37% ниже Welch.
- **Screening v3** (R = 200): MSE ниже Welch в 23 из 24 миров. Исключение —
  `gaussian_r2_00_n200` с отношением 1.06: плата за лишние параметры при нулевом сигнале.
- **Mini-check v3.1** (6 миров, R = 60, только как sanity):
  - в Freedman-мире pooled CUPED/CUPAC хуже Welch (×1.24–1.40), а Лин — нет (0.96);
  - в панельном мире CUPAC A ≈ Лин;
  - в нелинейном мире GBM ML-RATE лучше Лина (0.61 против 0.68).

**Заранее зафиксированная гипотеза:** Lin+HC2 входит в топ-2 по MSE во всех классах
линейных миров, а GBM-ML-RATE обгоняет его в нелинейных.

---

## 4. Задача: полный пересчёт

Пошаговые команды — в `HOME_RUN.md`, выполнять ровно их. Кратко:

| Стадия | Что | R | Seed | Оценка CPU-ч |
|---|---|---|---|---|
| 0 | `pytest -q` (ожидается 205 passed) + smoke | — | 20261001 | ~0.05 |
| S1 screening | 37 миров × 16 пайплайнов (`configs/rnd_v3_1_screening_pipelines.txt`), эффекты 0/0.1/0.2 | 500 | 20261010 | ~12 |
| C1 confirmation | финалисты из `results/rnd_v3_1/screening/run_metadata.json`, все миры | 2000 | 20261020 | ~20–30 |
| P1 power | 8 миров из `HOME_RUN.md` §5, эффекты 0–0.3 | 1000 | 20261030 | ~3–5 |

Правила исполнения:

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` — иначе результаты
  перестанут быть бит-воспроизводимыми.
- Один процесс на мир (`--world`), параллельность `P = min(ядра, RAM_ГБ/2)`
  (около 1.8 ГБ на процесс), `--resume` для перезапусков.
- Все шарды одной стадии запускать **одним и тем же коммитом**: merge сверяет git SHA
  и хэш незакоммиченных изменений `src/`. Не править код между шардами одной стадии.
- После каждой стадии выполнить merge (`examples/merge_rnd_shards.py`) и проверить, что
  он прошёл без ошибок, а в `run_metadata.json` есть `finalists` и `selection_rule`.
- Не менять пороги приёмки и список миров/пайплайнов **после** просмотра результатов.
  Если изменение необходимо, зафиксируйте его как отдельную версию плана и объясните
  причину в отчёте.

### Критерии приёмки (зафиксированы, менять нельзя)

- Размер теста относительно истинного estimand: верхняя граница Уилсона FPR ≤ 0.07.
- Покрытие: нижняя граница Уилсона ≥ 0.93.
- Смещение: |b| + 1.96·mcse ≤ 0.10·SD.
- SE/SD: нижняя граница CI ≥ 0.90; значение > 1.10 — флаг `conservative`, не провал.
- Super-uniformity p-value: не отвергнута после BH (q = 0.05) по стадии.
- **Выбор победителя.** Среди валидных во всех мирах своего класса сравнивается средний
  log(MSE / MSE Welch) по классам миров. Миры-дубли не считаются дважды:
  `strong_prior_mismatch ≡ gaussian_r2_04_n2000` и `panel_shift ≡ temporal_drift`
  для identity-пайплайнов. Победитель должен опережать второго по парной MSE с 95% CI,
  не содержащим 0, в большинстве классов. Иначе — ничья, и решает медиана runtime.

---

## 5. Анализ после расчёта (что нужно получить)

1. **Таблица валидности** по каждой паре «процедура × мир»: FPR против truth,
   покрытие, SE/SD, bias ± MC SE, вердикт и причины (`verdict_reasons`).
   Отдельно перечислить провалы.
2. **Таблица эффективности** по классам миров: MSE/MSE_Welch с CI парной разности,
   снижение дисперсии, мощность и MDE80 (из P1).
3. **Вердикт по гипотезе §3** — подтверждена или нет, с числами.
4. **Ответы на конкретные вопросы:**
   - выигрывает ли GBM-ML-RATE у Лина только в нелинейных мирах и какой ценой по времени;
   - становятся ли CUPED/CUPAC (pooled-θ) конкурентами Лина в панельных мирах;
   - подтверждается ли проигрыш pooled CUPED/ANCOVA разности средних в Freedman-мирах;
   - как ведёт себя Лин при малом n (n = 200 и n = 500) и в high-dim; не нужен ли HC3 по умолчанию;
   - PATE-трек: `lin_regression__hc2_pate` намеренно проваливается в `population_shift`
     (когорта не iid-выборка из популяции) — подтвердить, что вердикт `bad` и причина
     указаны верно.
5. **Обновить PDF-отчёт по лидеру** (или по новому лидеру, если он сменится):
   - заменить `docs/reports/lin_hc2/tables/confirmation_identity.csv` и
     `allworlds_screening_lin.csv` данными C1 (структура скрипта извлечения — в §8);
   - добавить в `report.tex` строки CUPED/CUPAC, которые теперь валидны, и раздел
     с кривыми мощности;
   - убрать оговорки «до исправлений»;
   - пересобрать: `docs/reports/lin_hc2/build.sh` (нужен
     [tectonic](https://tectonic-typesetting.github.io); путь задаётся `TECTONIC=...`).

---

## 6. Что вернуть пользователю

- Коммит в `rerun/v3.1` с `results/rnd_v3_1/{screening,confirmation,power}/`.
  Каталог `shards/` не коммитить; parquet файлы `pipeline_matrix` должны остаться
  в `.gitignore`.
- Обновлённый PDF `outputs/reports/VarWar_Lin_HC2_report_ru.pdf`.
- Короткое резюме на русском:
  - кто победил в каждом треке;
  - подтвердилась ли гипотеза;
  - что не прошло валидность и почему;
  - что рекомендуется для продакшена.
- Список отклонений от плана, если они были.

---

## 7. Известные ограничения и открытые задачи (не исправлены, приоритет — по ситуации)

| Пункт | Где | Суть |
|---|---|---|
| PATE-вариант Лина | `estimators.py`, `lin_regression__hc2_pate` | Предполагает, что когорта — iid-выборка из популяции. Не использует известные популяционные моменты (реестр их получает) и не учитывает поправку (1 − n/N). Для `population_shift` нужен вариант с центрированием на $\mu^*$. |
| GBM ML-RATE без категорий | `estimators.py` ~464, `analysis_registry.py` ~230 | `include_categorical=False`: `region` и `device` не используются. |
| Bell–McCaffrey df | `estimators.py` | Опция реализована и проверена, но ни один метод гонки её не использует. |
| Группы без строки | `rnd_runner.py` ~1420 | Если все эффекты группы NaN без execution failure (например, нет truth у `common_effect` или isolation-forest estimand), агрегатная строка не создаётся. |
| Покрытие миров финалистом | `select_confirmation_finalists` | `n_worlds` и `n_worlds_in_track` пишутся в metadata, но неполное покрытие не исключает пайплайн. Проверять вручную. |
| Inference при rerandomization | `inference.py` | Не реализован (план P2-4). Design race D1 не проводилась. |
| Resampling-inference в полной матрице | — | Не входит в план: около 10 тыс. CPU-ч. Сначала пруннинг по итогам S1. |
| Реальные данные | — | Адаптера нет. Кандидаты для полусинтетики: X5 RetailHero, Hillstrom, Lenta (`sklift.datasets.fetch_*`). Только публичные данные, ничего в git. |
| Многопоточность | — | При `OMP_NUM_THREADS > 1` Лин даёт расхождения около 4e-16 между запусками. |

Открытые вопросы к пользователю, если понадобится уточнение:

- основной estimand — SATE или PATE;
- нужны ли в отчёте все четыре контекста доступа;
- включать ли design race;
- HC2 или HC3 по умолчанию при малых n;
- типичные n, доли и типы метрик на реальных данных.

---

## 8. Карта репозитория (ключевое)

```
src/varwar/synthetic_v3.py     миры (37), spec_for_estimand, world_*_cols
src/varwar/rnd_runner.py       стадии, seed, broadcast, агрегирование, финалисты
src/varwar/validation.py       метрики и вердикты (MC-aware)
src/varwar/estimators.py       Lin (RegressionAdjustmentEstimator), ML-RATE (ridge/GBM), прочее
src/varwar/inference.py        Welch/Student/normal/permutation/bootstrap-t (materialized target)
src/varwar/analysis_registry.py, method_catalog.py   регистрация методов (синхронно с docs/methods/all_methods.md)
src/varwar/target_registry.py, transforms.py          CUPED/CUPAC A/B/C, pooled-θ
examples/run_rnd_v3.py         CLI стадий и шардов (--help)
examples/merge_rnd_shards.py   merge с проверками
configs/rnd_v3_1_screening_pipelines.txt              список пайплайнов S1
docs/review/v3_review_design.md                       полное ревью v3 и план
docs/methods/statistical_validation.md                правила вердиктов
docs/reports/lin_hc2/          отчёт по Лину: report.tex, sim_theory.py, make_figures.py, tables/, build.sh
HOME_RUN.md                    команды полного расчёта
```

Пример извлечения таблицы для отчёта (адаптировать к C1):

```python
import pandas as pd
a = pd.read_parquet("results/rnd_v3_1/confirmation/aggregates.parquet")
a = a[(a.information_mode == "experiment_only") & (a.temporal_access == "pre_treatment_only")
      & (a.ranking_estimand_id == "sample_ate")]
cols = ["world_id", "pipeline", "effect_size", "size", "size_ci_high", "size_metric", "h1_coverage", "h1_bias",
        "h1_bias_mc_se", "h1_se_to_sd", "h1_rmse", "rmse_ratio_vs_welch_cr", "power",
        "mde_80", "inference_quality", "verdict_reasons"]
```
