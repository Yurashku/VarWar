"""Иллюстративные Monte-Carlo для отчёта по регрессии Лина (HC2).

Самодостаточный скрипт (numpy/scipy), не зависит от стенда VarWar, чтобы
отдельно проверить теоретические утверждения отчёта:

1. Тождества: закрытая форма Лина = OLS-коэффициент; HC2 без ковариат = Welch.
2. Кейс Фридмана: неравный сплит + разные наклоны в плечах. ANCOVA/CUPED с общим
   наклоном может проигрывать разности средних, Лин — нет.
3. SATE vs PATE: HC2 консервативна для SATE, но антиконсервативна для PATE при
   гетерогенных наклонах; поправка gamma' Sigma_X gamma / n это исправляет.
4. Малые выборки: HC2 vs HC3, квантиль t_{n-p}.

Запуск:  python docs/reports/lin_hc2/sim_theory.py  (≈1–2 мин)
Выход:   docs/reports/lin_hc2/tables/sim_*.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "tables"
SEED = 20261020


def ols_hc(X: np.ndarray, y: np.ndarray, kind: str = "HC2"):
    """OLS с робастной ковариацией HC0/HC2/HC3. Возвращает (beta, V)."""
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    if kind == "HC0":
        w = e**2
    elif kind == "HC2":
        w = e**2 / (1.0 - h)
    elif kind == "HC3":
        w = e**2 / (1.0 - h) ** 2
    else:
        raise ValueError(kind)
    meat = (X * w[:, None]).T @ X
    return beta, xtx_inv @ meat @ xtx_inv


def lin(y, t, x, kind="HC2", pate=False):
    """Оценка Лина: y ~ 1 + t + (x - x̄) + t·(x - x̄). Возвращает (tau, se, df)."""
    xc = x - x.mean(axis=0)
    X = np.column_stack([np.ones_like(y), t, xc, t[:, None] * xc])
    beta, V = ols_hc(X, y, kind)
    k = x.shape[1]
    var = V[1, 1]
    if pate:
        gamma = beta[2 + k : 2 + 2 * k]
        sigma = np.atleast_2d(np.cov(x, rowvar=False))
        var = var + gamma @ sigma @ gamma / len(y)
    return beta[1], np.sqrt(var), len(y) - X.shape[1]


def ancova(y, t, x, kind="HC2"):
    xc = x - x.mean(axis=0)
    X = np.column_stack([np.ones_like(y), t, xc])
    beta, V = ols_hc(X, y, kind)
    return beta[1], np.sqrt(V[1, 1]), len(y) - X.shape[1]


def welch(y, t):
    y1, y0 = y[t == 1], y[t == 0]
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    df = (v1 + v0) ** 2 / (v1**2 / (len(y1) - 1) + v0**2 / (len(y0) - 1))
    return y1.mean() - y0.mean(), np.sqrt(v1 + v0), df


def cuped_pooled(y, t, x):
    """CUPED с θ по всей выборке + Welch на скорректированном исходе."""
    x1 = x[:, 0]
    theta = np.cov(y, x1)[0, 1] / np.var(x1, ddof=1)
    return welch(y - theta * (x1 - x1.mean()), t)


def complete_randomization(rng, n, p):
    t = np.zeros(n)
    t[rng.choice(n, size=int(round(p * n)), replace=False)] = 1.0
    return t


def check_identities(rng):
    n, p = 300, 0.3
    x = rng.normal(size=(n, 2))
    t = complete_randomization(rng, n, p)
    y = 1 + 0.5 * t + x @ np.array([1.0, -0.5]) + t * x[:, 0] + rng.normal(size=n)
    tau, _, _ = lin(y, t, x)
    xb = x.mean(axis=0)
    b1 = np.linalg.lstsq(np.column_stack([np.ones((t == 1).sum()), x[t == 1]]), y[t == 1], rcond=None)[0]
    b0 = np.linalg.lstsq(np.column_stack([np.ones((t == 0).sum()), x[t == 0]]), y[t == 0], rcond=None)[0]
    closed = (b1[0] + xb @ b1[1:]) - (b0[0] + xb @ b0[1:])
    _, se_hc2_nocov, _ = (lambda b, V: (b[1], np.sqrt(V[1, 1]), None))(*ols_hc(np.column_stack([np.ones(n), t]), y, "HC2"))
    _, se_w, _ = welch(y, t)
    return {"lin_minus_closed_form": abs(tau - closed), "hc2_nocov_minus_welch": abs(se_hc2_nocov - se_w)}


def mc(name, gen, methods, reps, rng):
    rows = {m: {"est": [], "se": [], "cover_s": [], "cover_p": [], "rej0": []} for m in methods}
    for _ in range(reps):
        y, t, x, sate, pate = gen(rng)
        for m, f in methods.items():
            est, se, df = f(y, t, x)
            q = stats.t.ppf(0.975, df)
            r = rows[m]
            r["est"].append(est - sate)
            r["se"].append(se)
            r["cover_s"].append(abs(est - sate) <= q * se)
            r["cover_p"].append(abs(est - pate) <= q * se)
            r["rej0"].append(abs(est) > q * se)
    out = []
    base_var = np.var(rows[next(iter(methods))]["est"], ddof=1)
    for m, r in rows.items():
        err = np.array(r["est"])
        sd = err.std(ddof=1)
        cs, cp = np.mean(r["cover_s"]), np.mean(r["cover_p"])
        out.append({
            "scenario": name, "method": m, "reps": reps,
            "bias": err.mean(), "sd": sd, "mean_se": np.mean(r["se"]),
            "se_to_sd": np.mean(r["se"]) / sd,
            "var_rel_dim": sd**2 / base_var,
            "coverage_sate": cs, "coverage_pate": cp,
            "coverage_mcse": np.sqrt(cs * (1 - cs) / reps),
            "rejection_vs_0": np.mean(r["rej0"]),
        })
    return out


def gen_freedman(n=400, p=0.2, b0=1.0, b1=0.0, tau=0.0):
    """Неравный сплит, разные наклоны: Y(0)=b0·x+ε, Y(1)=τ+b1·x+ε. PATE = τ (E x = 0)."""
    def g(rng):
        x = rng.normal(size=(n, 1))
        y0 = b0 * x[:, 0] + rng.normal(size=n)
        y1 = tau + b1 * x[:, 0] + rng.normal(size=n)
        t = complete_randomization(rng, n, p)
        y = np.where(t == 1, y1, y0)
        return y, t, x, (y1 - y0).mean(), tau
    return g


def gen_small(n, p=0.3, k=5):
    def g(rng):
        x = rng.normal(size=(n, k))
        beta = np.r_[1.0, np.zeros(k - 1)]
        eps = rng.normal(size=n) * (0.5 + np.abs(x[:, 0]))  # гетероскедастичность
        y0 = x @ beta + eps
        y1 = y0 + 0.3 + 0.5 * x[:, 0]
        t = complete_randomization(rng, n, p)
        y = np.where(t == 1, y1, y0)
        return y, t, x, (y1 - y0).mean(), 0.3
    return g


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    import csv

    ident = check_identities(rng)
    with open(OUT / "sim_identities.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "abs_diff"])
        for k, v in ident.items():
            w.writerow([k, f"{v:.3e}"])

    methods = {
        "dim_welch": lambda y, t, x: welch(y, t),
        "cuped_pooled_welch": cuped_pooled,
        "ancova_hc2": ancova,
        "lin_hc2": lin,
        "lin_hc2_pate": lambda y, t, x: lin(y, t, x, pate=True),
    }
    rows = []
    rows += mc("freedman_p02_b1_0", gen_freedman(), methods, 4000, rng)
    rows += mc("freedman_p05_b1_0", gen_freedman(p=0.5), methods, 4000, rng)
    rows += mc("homogeneous_p02", gen_freedman(b1=1.0), methods, 4000, rng)
    small = {
        "dim_welch": lambda y, t, x: welch(y, t),
        "lin_hc2": lin,
        "lin_hc3": lambda y, t, x: lin(y, t, x, kind="HC3"),
    }
    for n in (40, 80, 200, 1000):
        rows += mc(f"small_n{n}_p03_k5", gen_small(n), small, 4000, rng)
    keys = list(rows[0].keys())
    with open(OUT / "sim_mc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(ident)
    for r in rows:
        print(f"{r['scenario']:22s} {r['method']:20s} var_rel={r['var_rel_dim']:.3f} se/sd={r['se_to_sd']:.3f} "
              f"covS={r['coverage_sate']:.3f} covP={r['coverage_pate']:.3f}")


if __name__ == "__main__":
    main()
