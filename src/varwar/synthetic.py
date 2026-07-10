from __future__ import annotations

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


def generate_synthetic_ab(config: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    """Generate one two-arm experiment with covariates and auxiliary metrics."""

    rng = np.random.default_rng(config.seed)
    n = config.n
    treatment = rng.binomial(1, 0.5, size=n)
    y_pre = rng.normal(0.0, 1.0, size=n)
    x1 = config.covariate_signal * y_pre + rng.normal(0.0, 0.7, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    calendar_index = rng.normal(0.0, 1.0, size=n)
    market_index = 0.5 * calendar_index + rng.normal(0.0, 0.6, size=n)
    operational_load = 0.35 * y_pre + 0.4 * calendar_index + rng.normal(0.0, 0.7, size=n)
    if config.heavy_tail:
        noise = rng.standard_t(df=3, size=n) * config.noise_scale
    else:
        noise = rng.normal(0.0, config.noise_scale, size=n)

    baseline = (
        2.0
        + config.covariate_signal * y_pre
        + 0.35 * x1
        - 0.15 * x2
        + 0.20 * calendar_index
        + 0.18 * market_index
        + 0.25 * operational_load
    )
    y0 = baseline + noise
    y = y0 + treatment * config.effect

    denominator = rng.poisson(lam=4.0, size=n).astype(float) + 1.0
    numerator_rate = 0.2 + 0.02 * baseline + treatment * config.effect / 20.0
    numerator = denominator * numerator_rate + rng.normal(0.0, 0.2, size=n)

    takeup_prob = np.clip(0.08 + 0.75 * treatment + 0.08 * y_pre, 0.0, 1.0)
    received = rng.binomial(1, takeup_prob, size=n)
    y_compliance = y0 + received * config.effect
    post_outcome_proxy = y + rng.normal(0.0, 0.05, size=n)
    strata = np.where(y_pre < np.quantile(y_pre, 0.33), "low", "mid")
    strata = np.where(y_pre > np.quantile(y_pre, 0.66), "high", strata)

    return pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "treatment": treatment,
            "y": y,
            "y0": y0,
            "y_pre": y_pre,
            "x1": x1,
            "x2": x2,
            "calendar_index": calendar_index,
            "market_index": market_index,
            "operational_load": operational_load,
            "post_outcome_proxy": post_outcome_proxy,
            "numerator": numerator,
            "denominator": denominator,
            "assigned": treatment,
            "received": received,
            "y_compliance": y_compliance,
            "stratum": strata,
        }
    )


def generate_historical_controls(config: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    historical = generate_synthetic_ab(
        SyntheticConfig(
            n=config.n,
            effect=0.0,
            covariate_signal=config.covariate_signal,
            noise_scale=config.noise_scale,
            seed=config.seed + 10_000,
            heavy_tail=config.heavy_tail,
        )
    )
    historical["treatment"] = 0
    return historical
