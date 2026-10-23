"""Рисунки для отчёта по регрессии Лина (векторный PDF, matplotlib).

Читает tables/allworlds_screening_lin.csv и tables/sim_mc.csv.
Запуск: python docs/reports/lin_hc2/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
TAB, FIG = HERE / "tables", HERE / "figures"

# Референсная палитра (светлый режим), слоты 1–3 в фиксированном порядке.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42,
})

WORLD_RU = {
    "gaussian_r2_00_n200": "Гаусс, R²=0, n=200", "gaussian_r2_00_n2000": "Гаусс, R²=0, n=2000",
    "gaussian_r2_04_n200": "Гаусс, R²=0.4, n=200", "gaussian_r2_04_n2000": "Гаусс, R²=0.4, n=2000",
    "gaussian_r2_08_n200": "Гаусс, R²=0.8, n=200", "gaussian_r2_08_n2000": "Гаусс, R²=0.8, n=2000",
    "gaussian_30_70": "Сплит 30/70", "student_t3": "Стьюдент t₃", "lognormal": "Логнормальный",
    "contaminated": "Засорение 3%", "zero_inflated_count": "Счётчик с нулями",
    "heteroskedastic": "Гетероскедастичность", "quadratic_signal": "Квадратичный сигнал",
    "threshold_signal": "Пороговый сигнал", "interactions_high_dimensional": "Взаимодействия + 24 шум. признака",
    "heterogeneous_effect": "Гетерогенный эффект", "population_shift": "Когорта ≠ популяция",
    "panel_shift": "Сдвиг исторической панели", "temporal_drift": "Дрейф во времени",
    "strong_prior_mismatch": "Неверная историческая оценка", "temporal_bc_safe": "Признаки во время теста, чистые",
    "temporal_b_unsafe": "Признак зависит от воздействия", "temporal_c_unsafe": "Признак — прокси исхода",
    "many_rare_strata": "Много редких страт",
}


def fig_allworlds():
    d = pd.read_csv(TAB / "allworlds_screening_lin.csv").sort_values("mse_lin_rel", ascending=False)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    ax.axvline(1.0, color=INK2, lw=1, ls=(0, (3, 2)), zorder=1)
    ax.scatter(d.mse_ml_rel, y + 0.18, s=26, color=ORANGE, label="ML-RATE (линейный прогноз) + HC2", zorder=3,
               edgecolor="white", linewidth=1.2)
    ax.scatter(d.mse_lin_rel, y - 0.18, s=26, color=BLUE, label="Лин + HC2", zorder=3, edgecolor="white", linewidth=1.2)
    ax.set_yticks(y, [WORLD_RU.get(w, w) for w in d.world_id])
    ax.set_xlabel("MSE / MSE разности средних (меньше — лучше)")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "allworlds_mse.pdf")
    plt.close(fig)


def fig_freedman():
    s = pd.read_csv(TAB / "sim_mc.csv")
    order = ["dim_welch", "cuped_pooled_welch", "ancova_hc2", "lin_hc2"]
    lab = {"dim_welch": "Разность средних (Уэлч)", "cuped_pooled_welch": "CUPED (общий θ)",
           "ancova_hc2": "ANCOVA (общий наклон)", "lin_hc2": "Лин (взаимодействия)"}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2), sharey=True)
    for ax, sc, title in zip(axes, ["freedman_p02_b1_0", "freedman_p05_b1_0"], ["p = 0.2", "p = 0.5"]):
        r = s[s.scenario == sc].set_index("method").loc[order]
        yy = np.arange(len(order))[::-1]
        colors = [INK2, ORANGE, ORANGE, BLUE]
        ax.barh(yy, r.var_rel_dim, color=colors, height=0.55)
        for yi, v in zip(yy, r.var_rel_dim):
            ax.text(v + 0.02, yi, f"{v:.2f}", va="center", color=INK, fontsize=8)
        ax.axvline(1.0, color=INK2, lw=1, ls=(0, (3, 2)))
        ax.set_title(f"Неравные наклоны, {title}", fontsize=9, color=INK)
        ax.set_xlim(0, 1.65)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(yy, [lab[m] for m in order])
        ax.set_xlabel("Var(τ̂) / Var(τ̂ разности средних)")
    fig.tight_layout()
    fig.savefig(FIG / "freedman.pdf")
    plt.close(fig)


def fig_small_n():
    s = pd.read_csv(TAB / "sim_mc.csv")
    s = s[s.scenario.str.startswith("small_")].copy()
    s["n"] = s.scenario.str.extract(r"n(\d+)").astype(int)
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    ax.axhspan(0.94, 0.96, color=GRID, alpha=0.6, lw=0)
    ax.axhline(0.95, color=INK2, lw=1, ls=(0, (3, 2)))
    for m, c, lab, dx in [("dim_welch", INK2, "Уэлч", -0.03), ("lin_hc2", BLUE, "Лин + HC2", 0.0),
                          ("lin_hc3", AQUA, "Лин + HC3", 0.03)]:
        r = s[s.method == m].sort_values("n")
        x = np.log10(r.n) + dx
        ax.errorbar(x, r.coverage_sate, yerr=1.96 * r.coverage_mcse, color=c, marker="o", ms=4.5, lw=2,
                    capsize=0, label=lab)
    ax.set_xticks(np.log10([40, 80, 200, 1000]), ["40", "80", "200", "1000"])
    ax.set_xlim(1.5, 3.1)
    ax.set_ylim(0.915, 0.995)
    ax.set_xlabel("n (сплит 30/70, K = 5 ковариат, гетероскедастичность)")
    ax.set_ylabel("Покрытие 95% CI")
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "small_n.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_allworlds()
    fig_freedman()
    fig_small_n()
    print("ok")
