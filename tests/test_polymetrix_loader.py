"""Tests for the PolyMetriX Tg loader."""

from __future__ import annotations

import pandas as pd
import pytest

from descriptor_limitations.data_loaders import (
    expand_polymetrix_tg,
    load_polymetrix_tg,
)


@pytest.fixture(scope="module")
def tg_df() -> pd.DataFrame:
    return load_polymetrix_tg()


def test_polytg_shape(tg_df: pd.DataFrame) -> None:
    assert len(tg_df) == 7367


def test_polytg_reliability_counts(tg_df: pd.DataFrame) -> None:
    counts = tg_df["meta.reliability"].value_counts().to_dict()
    assert counts == {"black": 7088, "gold": 143, "yellow": 132, "red": 4}


def test_polytg_required_columns(tg_df: pd.DataFrame) -> None:
    for col in [
        "PSMILES", "labels.Exp_Tg(K)",
        "meta.tg_values", "meta.num_of_points",
        "meta.std", "meta.reliability",
    ]:
        assert col in tg_df.columns


def test_expand_total_rows(tg_df: pd.DataFrame) -> None:
    """Sum of num_of_points across all rows = total measurements."""
    long = expand_polymetrix_tg(tg_df)
    expected = int(tg_df["meta.num_of_points"].sum())
    assert len(long) == expected


def test_expand_singletons_preserve_value(tg_df: pd.DataFrame) -> None:
    """Black-tier rows (num_of_points=1) -> single expanded row with the
    curated Exp_Tg(K) value."""
    sample = tg_df[tg_df["meta.num_of_points"] == 1].head(5)
    long = expand_polymetrix_tg(sample)
    assert len(long) == 5
    assert (long["measurement_index"] == 0).all()
    for _, row in sample.iterrows():
        ex = long[long["psmiles"] == row["PSMILES"]]
        assert ex["tg_K"].iloc[0] == pytest.approx(row["labels.Exp_Tg(K)"])


def test_expand_multisource_matches_count(tg_df: pd.DataFrame) -> None:
    """Yellow/Gold rows expand to num_of_points rows each."""
    sample = tg_df[tg_df["meta.reliability"].isin(["yellow", "gold"])].head(5)
    long = expand_polymetrix_tg(sample)
    grouped = long.groupby("psmiles").size()
    for psmiles, n in zip(sample["PSMILES"], sample["meta.num_of_points"]):
        assert grouped[psmiles] == n


def test_expand_long_has_no_nans(tg_df: pd.DataFrame) -> None:
    long = expand_polymetrix_tg(tg_df)
    assert long["tg_K"].notna().all()
    assert long["psmiles"].notna().all()
