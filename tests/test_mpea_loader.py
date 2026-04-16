"""Tests for the MPEA loader."""

from __future__ import annotations

import pandas as pd
import pytest

from descriptor_limitations.data_loaders import load_mpea


@pytest.fixture(scope="module")
def mpea_df() -> pd.DataFrame:
    return load_mpea()


def test_mpea_shape(mpea_df: pd.DataFrame) -> None:
    assert len(mpea_df) == 1545
    assert mpea_df["formula"].nunique() == 630


def test_mpea_key_coverage(mpea_df: pd.DataFrame) -> None:
    assert mpea_df["YS_MPa"].notna().sum() == 1067
    assert mpea_df["grain_size_um"].notna().sum() == 237


def test_mpea_hall_petch_subset(mpea_df: pd.DataFrame) -> None:
    """207 rows have both YS and grain size -- the Hall-Petch subset."""
    both = mpea_df[mpea_df["YS_MPa"].notna() & mpea_df["grain_size_um"].notna()]
    assert len(both) == 207


def test_mpea_short_aliases(mpea_df: pd.DataFrame) -> None:
    for c in [
        "formula", "microstructure", "processing", "phase",
        "grain_size_um", "YS_MPa", "UTS_MPa", "elongation_pct",
        "hardness_HV", "test_type", "test_T_C",
    ]:
        assert c in mpea_df.columns
