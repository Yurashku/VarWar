from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from varwar.core import ExperimentSpec
from varwar.method_catalog import (
    MethodCategory,
    MethodStatus,
    get_method_record,
    method_catalog,
)
from varwar.registry import (
    default_estimators,
    placeholder_design_preprocessors,
    placeholder_estimators,
    placeholder_metric_transformers,
    placeholder_target_transformers,
)


class MethodCatalogTests(unittest.TestCase):
    def test_catalog_ids_are_unique_and_default_race_is_covered(self) -> None:
        records = method_catalog()
        ids = {record.method_id for record in records}
        self.assertEqual(len(ids), len(records))
        race_ids = {estimator.name for estimator in default_estimators()}
        self.assertEqual(race_ids - ids, set())

    def test_selected_article_candidates_have_correct_roles(self) -> None:
        nonlinear = get_method_record("nonlinear_cupac")
        self.assertEqual(nonlinear.category, MethodCategory.TARGET_TRANSFORMER)
        self.assertEqual(nonlinear.status, MethodStatus.IMPLEMENTED)

        randomization = get_method_record("prognostic_score_block_randomization")
        self.assertEqual(
            randomization.category,
            MethodCategory.RANDOMIZATION_STRUCTURE_BUILDER,
        )
        self.assertEqual(randomization.status, MethodStatus.IMPLEMENTED)

        ratio = get_method_record("ratio_linearization")
        self.assertEqual(ratio.category, MethodCategory.METRIC_TRANSFORMER)
        self.assertEqual(ratio.status, MethodStatus.EXCLUDED)

    def test_all_placeholder_registries_are_non_executable_and_cataloged(self) -> None:
        data = pd.DataFrame({"treatment": [], "y": []})
        spec = ExperimentSpec()

        for placeholder in placeholder_estimators():
            self.assertFalse(get_method_record(placeholder.name).is_implemented)
            with self.assertRaises(NotImplementedError):
                placeholder.estimate(data, spec)

        for placeholder in placeholder_target_transformers():
            record = get_method_record(placeholder.name)
            self.assertEqual(record.category, MethodCategory.TARGET_TRANSFORMER)
            self.assertFalse(record.is_implemented)
            with self.assertRaises(NotImplementedError):
                placeholder.fit_transform(data, spec)

        for placeholder in placeholder_design_preprocessors():
            record = get_method_record(placeholder.name)
            self.assertEqual(record.category, MethodCategory.DESIGN_PREPROCESSOR)
            self.assertFalse(record.is_implemented)
            with self.assertRaises(NotImplementedError):
                placeholder.fit(data)

        for placeholder in placeholder_metric_transformers():
            record = get_method_record(placeholder.name)
            self.assertEqual(record.category, MethodCategory.METRIC_TRANSFORMER)
            self.assertFalse(record.is_implemented)
            with self.assertRaises(NotImplementedError):
                placeholder.fit_transform(data, spec)

    def test_markdown_registry_mentions_every_method_id(self) -> None:
        markdown = (ROOT / "docs" / "methods" / "all_methods.md").read_text(
            encoding="utf-8"
        )
        missing = [
            record.method_id
            for record in method_catalog()
            if f"`{record.method_id}`" not in markdown
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
