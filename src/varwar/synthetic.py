from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n: int = 5000
    effect: float = 0.25
    covariate_signal: float = 0.75
    noise_scale: float = 1.0
    seed: int = 42
    heavy_tail: bool = False
    noise_distribution: str | None = None
    student_t_df: float = 3.0
    contamination_probability: float = 0.02
    contamination_scale: float = 12.0
    outcome_model: str = "linear"
    effect_heterogeneity: float = 0.0
    expert_safe_treatment_shift: float = 0.0
    gated_treatment_shift: float = 0.0
    experiment_stratum_weights: tuple[float, float, float] | None = None


def generate_synthetic_panel(
    config: SyntheticConfig = SyntheticConfig(),
    *,
    n_months: int = 6,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Generate treatment-free monthly panel rows for panel-prior R&D.

    Every month is generated independently with the same schema as the
    experiment.  The function is intentionally a local synthetic helper; a
    production caller should provide ``PanelPriorSummary`` from SQL instead
    of passing panel rows into an estimator.
    """

    if n_months < 2:
        raise ValueError("n_months must be at least two for temporal validation.")
    frames: list[pd.DataFrame] = []
    dates = pd.date_range(start=start_date, periods=n_months, freq="MS")
    for month_index, date in enumerate(dates):
        month_config = SyntheticConfig(
            n=config.n,
            effect=0.0,
            covariate_signal=config.covariate_signal,
            noise_scale=config.noise_scale,
            seed=config.seed + 50_000 + month_index,
            heavy_tail=config.heavy_tail,
            noise_distribution=config.noise_distribution,
            student_t_df=config.student_t_df,
            contamination_probability=config.contamination_probability,
            contamination_scale=config.contamination_scale,
            outcome_model=config.outcome_model,
            effect_heterogeneity=config.effect_heterogeneity,
            experiment_stratum_weights=config.experiment_stratum_weights,
        )
        month = generate_synthetic_ab(month_config)
        month["treatment"] = 0
        month["report_date"] = date
        month["entity_id"] = month["unit_id"]
        month["panel_month"] = month_index
        frames.append(month)
    return pd.concat(frames, ignore_index=True)


def generate_synthetic_ab(config: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    """Generate one two-arm experiment with covariates and auxiliary metrics."""

    rng = np.random.default_rng(config.seed)
    n = config.n
    treatment = rng.binomial(1, 0.5, size=n)
    if config.outcome_model not in {"linear", "nonlinear"}:
        raise ValueError("outcome_model must be 'linear' or 'nonlinear'.")
    y_pre_lag2 = rng.normal(0.0, 1.0, size=n)
    x1_lag2 = config.covariate_signal * y_pre_lag2 + rng.normal(0.0, 0.7, size=n)
    x2_lag2 = rng.normal(0.0, 1.0, size=n)
    calendar_index_lag2 = rng.normal(0.0, 1.0, size=n)
    market_index_lag2 = 0.5 * calendar_index_lag2 + rng.normal(0.0, 0.6, size=n)
    operational_load_lag2 = (
        0.35 * y_pre_lag2
        + 0.4 * calendar_index_lag2
        + rng.normal(0.0, 0.7, size=n)
    )
    lag_correlation = 0.65
    y_pre_signal = (
        lag_correlation * y_pre_lag2
        + 0.18 * x1_lag2
        - 0.08 * x2_lag2
        + 0.10 * calendar_index_lag2
    )
    if config.outcome_model == "nonlinear":
        y_pre_signal += (
            0.30 * (y_pre_lag2**2 - 1.0)
            + 0.20 * np.sin(x2_lag2)
            + 0.12 * x1_lag2 * calendar_index_lag2
        )
    y_pre = y_pre_signal + math.sqrt(1.0 - lag_correlation**2) * rng.normal(
        0.0,
        1.0,
        size=n,
    )
    x1 = config.covariate_signal * y_pre + rng.normal(0.0, 0.7, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    calendar_latent = rng.normal(0.0, 1.0, size=n)
    market_latent = 0.5 * calendar_latent + rng.normal(0.0, 0.6, size=n)
    operational_latent = (
        0.35 * y_pre + 0.4 * calendar_latent + rng.normal(0.0, 0.7, size=n)
    )
    noise = _generate_noise(rng, config, n)

    population_stratum = None
    population_score = np.zeros(n, dtype=float)
    if config.experiment_stratum_weights is not None:
        probabilities = np.asarray(config.experiment_stratum_weights, dtype=float)
        if probabilities.shape != (3,) or np.any(probabilities < 0.0):
            raise ValueError("experiment_stratum_weights must contain three non-negative values.")
        if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("experiment_stratum_weights must sum to one.")
        population_stratum = rng.choice(
            np.array(["low", "mid", "high"], dtype=object),
            size=n,
            p=probabilities,
        )
        population_score = np.select(
            [population_stratum == "low", population_stratum == "high"],
            [-1.0, 1.0],
            default=0.0,
        )

    baseline = (
        2.0
        + config.covariate_signal * y_pre
        + 0.35 * x1
        - 0.15 * x2
        + 0.20 * calendar_latent
        + 0.18 * market_latent
        + 0.25 * operational_latent
        + 0.60 * population_score
    )
    if config.outcome_model == "nonlinear":
        baseline += (
            0.40 * (y_pre**2 - float(np.mean(y_pre**2)))
            + 0.25 * np.sin(x2)
            + 0.15 * x1 * calendar_latent
        )
    y0 = baseline + noise
    centered_pre = y_pre - float(y_pre.mean())
    individual_effect = config.effect + config.effect_heterogeneity * centered_pre
    y = y0 + treatment * individual_effect

    calendar_index = (
        calendar_latent + treatment * config.expert_safe_treatment_shift
    )
    market_index = market_latent
    operational_load = (
        operational_latent + treatment * config.gated_treatment_shift
    )

    denominator = rng.poisson(lam=4.0, size=n).astype(float) + 1.0
    numerator_rate = 0.2 + 0.02 * baseline + treatment * config.effect / 20.0
    numerator = denominator * numerator_rate + rng.normal(0.0, 0.2, size=n)

    takeup_prob = np.clip(0.08 + 0.75 * treatment + 0.08 * y_pre, 0.0, 1.0)
    received = rng.binomial(1, takeup_prob, size=n)
    y_compliance = y0 + received * config.effect
    post_outcome_proxy = y + rng.normal(0.0, 0.05, size=n)
    strata = np.where(y_pre < np.quantile(y_pre, 0.33), "low", "mid")
    strata = np.where(y_pre > np.quantile(y_pre, 0.66), "high", strata)
    region = np.where(x2 < -0.5, "west", np.where(x2 > 0.5, "east", "central"))
    device = np.where(x1 < -0.4, "mobile", np.where(x1 > 0.6, "desktop", "tablet"))
    activity_band = np.where(
        y_pre < -0.7,
        "low",
        np.where(y_pre > 0.7, "high", "medium"),
    )

    result = pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "treatment": treatment,
            "y": y,
            "y0": y0,
            "individual_effect": individual_effect,
            "y_pre": y_pre,
            "y_pre__lag2": y_pre_lag2,
            "x1": x1,
            "x1__lag2": x1_lag2,
            "x2": x2,
            "x2__lag2": x2_lag2,
            "calendar_index": calendar_index,
            "calendar_index__lag2": calendar_index_lag2,
            "market_index": market_index,
            "market_index__lag2": market_index_lag2,
            "operational_load": operational_load,
            "operational_load__lag2": operational_load_lag2,
            "post_outcome_proxy": post_outcome_proxy,
            "numerator": numerator,
            "denominator": denominator,
            "assigned": treatment,
            "received": received,
            "y_compliance": y_compliance,
            "stratum": strata,
            "region": pd.Categorical(region),
            "device": pd.Categorical(device),
            "activity_band": pd.Categorical(activity_band),
        }
    )
    if population_stratum is not None:
        result["population_stratum"] = pd.Categorical(
            population_stratum,
            categories=["low", "mid", "high"],
        )
        result["population_score"] = population_score
    return result


def _generate_noise(
    rng: np.random.Generator,
    config: SyntheticConfig,
    n: int,
) -> np.ndarray:
    distribution = config.noise_distribution
    if distribution is None:
        distribution = "student_t" if config.heavy_tail else "normal"
    if distribution == "normal":
        base = rng.normal(0.0, 1.0, size=n)
    elif distribution == "student_t":
        if config.student_t_df <= 2.0:
            raise ValueError("student_t_df must exceed two for finite variance.")
        base = rng.standard_t(df=config.student_t_df, size=n)
        base /= math.sqrt(config.student_t_df / (config.student_t_df - 2.0))
    elif distribution == "centered_lognormal":
        sigma = 1.0
        raw = rng.lognormal(mean=0.0, sigma=sigma, size=n)
        mean = math.exp(0.5 * sigma**2)
        variance = (math.exp(sigma**2) - 1.0) * math.exp(sigma**2)
        base = (raw - mean) / math.sqrt(variance)
    elif distribution == "contaminated_normal":
        if not 0.0 <= config.contamination_probability < 1.0:
            raise ValueError("contamination_probability must be in [0, 1).")
        if config.contamination_scale <= 0.0:
            raise ValueError("contamination_scale must be positive.")
        contaminated = rng.random(n) < config.contamination_probability
        scales = np.where(contaminated, config.contamination_scale, 1.0)
        base = rng.normal(0.0, scales, size=n)
    else:
        raise ValueError(
            "noise_distribution must be normal, student_t, centered_lognormal, "
            "or contaminated_normal."
        )
    return np.asarray(base, dtype=float) * config.noise_scale


def generate_historical_controls(config: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    historical = generate_synthetic_ab(
        SyntheticConfig(
            n=config.n,
            effect=0.0,
            covariate_signal=config.covariate_signal,
            noise_scale=config.noise_scale,
            seed=config.seed + 10_000,
            heavy_tail=config.heavy_tail,
            noise_distribution=config.noise_distribution,
            student_t_df=config.student_t_df,
            contamination_probability=config.contamination_probability,
            contamination_scale=config.contamination_scale,
            outcome_model=config.outcome_model,
            effect_heterogeneity=config.effect_heterogeneity,
            expert_safe_treatment_shift=0.0,
            gated_treatment_shift=0.0,
            experiment_stratum_weights=config.experiment_stratum_weights,
        )
    )
    historical["treatment"] = 0
    return historical
