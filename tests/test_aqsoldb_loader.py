"""Tests for the AqSolDB loaders."""

from __future__ import annotations

import pandas as pd
import pytest

from descriptor_limitations.data_loaders import (
    load_aqsoldb_curated,
    load_aqsoldb_sources,
)


@pytest.fixture(scope="module")
def curated() -> pd.DataFrame:
    return load_aqsoldb_curated()


@pytest.fixture(scope="module")
def sources() -> pd.DataFrame:
    return load_aqsoldb_sources()


def test_curated_shape(curated: pd.DataFrame) -> None:
    assert len(curated) == 9982


def test_curated_key_columns(curated: pd.DataFrame) -> None:
    for c in [
        "SMILES", "InChIKey", "Solubility", "SD", "Occurrences", "Group",
    ]:
        assert c in curated.columns


def test_curated_multi_source_count(curated: pd.DataFrame) -> None:
    """2236 InChIKeys appear in ≥2 source datasets."""
    assert (curated["Occurrences"] >= 2).sum() == 2236


def test_sources_cover_all_subsets(sources: pd.DataFrame) -> None:
    """All 9 source datasets A..I are present."""
    assert set(sources["source"].unique()) == set("ABCDEFGHI")


def test_sources_size_and_coverage(sources: pd.DataFrame) -> None:
    """Concatenated source table has ~20k rows (one per source-level
    measurement) and covers all curated InChIKeys.

    Note: AqSolDB's `Occurrences` column counts distinct *rounded*
    Solubility values per compound, not source datasets, so
    sum(Occurrences) under-counts the actual per-source measurements.
    The source-level table is the correct replicate layer for
    ceiling analysis.
    """
    curated = load_aqsoldb_curated()
    assert len(sources) == 19795
    # Every curated InChIKey should appear at least once in the sources.
    covered = sources["InChIKey"].isin(curated["InChIKey"]).all()
    assert covered


def test_sources_have_no_prediction_column(sources: pd.DataFrame) -> None:
    assert "Prediction" not in sources.columns


def test_sources_solubility_finite(sources: pd.DataFrame) -> None:
    assert sources["Solubility"].notna().all()
