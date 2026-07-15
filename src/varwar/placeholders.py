from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from varwar.core import ExperimentSpec


@dataclass(frozen=True)
class PlaceholderTargetTransformer:
    """Non-executable target-transformer candidate kept in the R&D registry."""

    name: str
    reason: str

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        raise NotImplementedError(f"{self.name} is not implemented: {self.reason}")

    def diagnostics(self) -> Mapping[str, Any]:
        return {"status": "not_implemented", "reason": self.reason}


@dataclass(frozen=True)
class PlaceholderDesignPreprocessor:
    """Non-executable design candidate kept outside estimator races."""

    name: str
    reason: str

    def fit(
        self,
        reference_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
    ) -> "PlaceholderDesignPreprocessor":
        raise NotImplementedError(f"{self.name} is not implemented: {self.reason}")

    def transform(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError(f"{self.name} is not implemented: {self.reason}")

    def diagnostics(self) -> Mapping[str, Any]:
        return {"status": "not_implemented", "reason": self.reason}
