# Полный пересчёт VarWar R&D v3.1 на домашней машине

Этот файл описывает **долгий расчёт**, который намеренно не запускался на рабочей
машине. Всё, что предшествует ему (исправления стенда, тесты, smoke и mini-check),
уже сделано в ветке `rerun/v3.1`. Контекст и обоснование плана —
`docs/review/v3_review_design.md` (§C — исправления, §D — дизайн пересчёта).

## 0. Важно перед началом

- **Не делайте `git pull` в старом клоне VarWar.** История `main` была переписана
  (из неё удалены выводы ноутбуков с реальными данными). Старый клон нужно удалить и
  склонировать репозиторий заново, иначе старая история вернётся при push.
- Шарды должны работать **в один поток** (`OMP_NUM_THREADS=1` и т.д.), иначе
  результаты не будут бит-воспроизводимыми: многопоточный BLAS даёт расхождения около 1e-16.
- Все шарды одной стадии должны запускаться **одной и той же версией кода**:
  merge проверяет git SHA и хэш незакоммиченных изменений в `src/`.

## 1. Установка

```bash
git clone https://github.com/Yurashku/VarWar.git
cd VarWar
git checkout rerun/v3.1
python3.12 -m venv .venv            # или: uv venv .venv --python 3.12
.venv/bin/pip install -e ".[dev,ml]" # hypex 1.0.7, scikit-learn, pyarrow, psutil
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest -q        # ожидается 204 passed, около 2 минут
```

## 2. Быстрая проверка (около 1 минуты)

```bash
.venv/bin/python examples/run_rnd_v3.py --stage smoke --output-dir results/rnd_v3_1
```

## 3. Стадия S1 — screening (37 миров × 16 пайплайнов, R=500, эффекты 0/0.1/0.2)

Шардирование по мирам: один процесс на мир, параллельность `-P` подбирается по памяти
(около 1.8 ГБ RSS на процесс). Seed зависит только от `(seed, world, trial)`, поэтому
состав и порядок шардов на результат не влияют. `--resume` пропускает готовые шарды.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P=6   # число параллельных процессов: min(ядра, RAM_ГБ / 2)
.venv/bin/python examples/run_rnd_v3.py --list-worlds | tr ' ' '\n' | \
  xargs -P $P -I{} .venv/bin/python examples/run_rnd_v3.py --stage screening \
    --world {} --pipelines-file configs/rnd_v3_1_screening_pipelines.txt \
    --output-dir results/rnd_v3_1/shards/screening/{} --resume
.venv/bin/python examples/merge_rnd_shards.py results/rnd_v3_1/shards/screening/* \
  --stage screening --output results/rnd_v3_1/screening
```

Merge падает при дублирующихся ключах или расхождении конфигурации и кода между
шардами. Он пересчитывает агрегаты и поправку BH, а финалистов выбирает кодом и
записывает их в `results/rnd_v3_1/screening/run_metadata.json` (поля `finalists`,
`selection_rule`).

## 4. Стадия C1 — confirmation (финалисты, R=2000, новый seed)

```bash
.venv/bin/python examples/run_rnd_v3.py --list-worlds | tr ' ' '\n' | \
  xargs -P $P -I{} .venv/bin/python examples/run_rnd_v3.py --stage confirmation \
    --world {} --pipelines-file results/rnd_v3_1/screening/run_metadata.json \
    --output-dir results/rnd_v3_1/shards/confirmation/{} --resume
.venv/bin/python examples/merge_rnd_shards.py results/rnd_v3_1/shards/confirmation/* \
  --stage confirmation --output results/rnd_v3_1/confirmation
```

## 5. Стадия P1 — кривые мощности (8 репрезентативных миров, R=1000)

```bash
for w in gaussian_r2_04_n200 gaussian_r2_04_n2000 gaussian_n2000_panel nonlinear_panel \
         freedman_p20_panel heteroskedastic lognormal population_shift; do echo $w; done | \
  xargs -P $P -I{} .venv/bin/python examples/run_rnd_v3.py --stage power \
    --world {} --pipelines-file results/rnd_v3_1/confirmation/run_metadata.json \
    --output-dir results/rnd_v3_1/shards/power/{} --resume
.venv/bin/python examples/merge_rnd_shards.py results/rnd_v3_1/shards/power/* \
  --stage power --output results/rnd_v3_1/power
```

## 6. Ожидаемые затраты

Оценка по mini-check на рабочей машине: 6 миров × 60 реализаций × 2 эффекта за
9 минут, около 0.75 с CPU на одну единицу «мир × эффект × реализация» для 16 пайплайнов.
Самый дорогой пайплайн — GBM ML-RATE, около 0.3 с на вызов.

| Стадия | Единиц | CPU-часы (оценка) | Wall при P=6 |
|---|---|---|---|
| S1 screening | 37 × 3 × 500 ≈ 55 500 | ≈ 12 | ≈ 2–2.5 ч |
| C1 confirmation | 37 × 3 × 2000 ≈ 222 000 | ≈ 20–30 (зависит от того, попадёт ли GBM в финалисты) | ≈ 4–5 ч |
| P1 power | 8 × 6 × 1000 = 48 000 | ≈ 3–5 | ≈ 1 ч |

Итого одна ночь. Пиковая память — около 1.8 ГБ на процесс.

## 7. Критерии приёмки (зафиксированы до запуска)

Они реализованы в `src/varwar/validation.py` и описаны в
`docs/methods/statistical_validation.md`:

- размер теста относительно истинного estimand: верхняя граница Уилсона FPR ≤ 0.07;
- покрытие: нижняя граница Уилсона ≥ 0.93;
- смещение: |b| + 1.96·mcse ≤ 0.10·SD;
- SE/SD: нижняя граница CI ≥ 0.90; значение > 1.10 считается флагом «conservative», а не провалом;
- super-uniformity p-value: не отвергнута после BH (q = 0.05) по стадии.

**Выбор лучшего пайплайна.** Среди валидных во всех мирах своего класса сравнивается
средний log(MSE / MSE Welch); победитель должен обгонять второе место по парной MSE.

**Гипотеза, проверяемая заранее.** Lin+HC2 входит в топ-2 по MSE во всех классах
линейных миров, а GBM-ML-RATE обгоняет его в нелинейных.

## 8. После расчёта

1. Отчёт по стадии лежит в `results/rnd_v3_1/<stage>/report_ru.md`, Excel собирается так:
   `node examples/build_rnd_v3_workbook.mjs` (см. `examples/build_rnd_v3_workbook.ps1`).
2. Чтобы обновить PDF-отчёт по Лину, замените таблицы `docs/reports/lin_hc2/tables/`
   данными C1/P1 и выполните `docs/reports/lin_hc2/build.sh`
   (нужен [tectonic](https://tectonic-typesetting.github.io)).
3. Закоммитьте `results/rnd_v3_1/{screening,confirmation,power}` (без `shards/`) и
   пришлите их: я обновлю отчёт и выводы.

## 9. Реальные данные (опционально)

Синтетика покрывает 37 миров. Если понадобится полусинтетическая проверка на публичных
данных (случайное назначение и известный внедрённый эффект при реальном совместном
распределении признаков), подходят следующие наборы:

- **X5 RetailHero** (`sklift.datasets.fetch_x5`): 45.8 млн транзакций, из которых
  строится панель «предпериод → период исхода»; тяжёлые хвосты выручки.
- **Hillstrom** (MineThatData E-Mail Analytics, `sklift.datasets.fetch_hillstrom`):
  реальное RCT, `history` — расходы в предпериоде.
- **Lenta** (`sklift.datasets.fetch_lenta`): агрегаты до кампании, бинарный исход.

Локальные копии уже есть в `~/projects/rnd_reports/data/06_safe_intime_cupac/{x5,hillstrom,lenta}/`.
Адаптера к стенду v3 пока нет: это следующий шаг, если он нужен.
