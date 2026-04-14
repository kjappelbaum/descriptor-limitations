"""Loader tests.

These tests hit the network on first run (cached thereafter in the repo's
``data/`` directory). They are marked with ``network`` so they can be
deselected in offline CI if needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from descriptor_limitations.data_loaders import load_doyle


@pytest.fixture(scope="module")
def doyle_df() -> pd.DataFrame:
    return load_doyle()


def test_doyle_shape(doyle_df: pd.DataFrame) -> None:
    """4599 raw rows minus 287 halide-blank wells = 4312."""
    assert len(doyle_df) == 4312


def test_doyle_no_missing_halide(doyle_df: pd.DataFrame) -> None:
    assert doyle_df["halide"].notna().all()


def test_doyle_no_additive_means_none(doyle_df: pd.DataFrame) -> None:
    """NaN additives have been recoded to the string 'NONE'."""
    assert doyle_df["additive"].notna().all()
    # 192 rows in raw had NaN additive, but 12 of those also had NaN halide
    # and were dropped in the halide-blank filter. 192 - 12 = 180 remain.
    assert (doyle_df["additive"] == "NONE").sum() == 180


def test_doyle_factor_cardinalities(doyle_df: pd.DataFrame) -> None:
    assert doyle_df["base"].nunique() == 3
    assert doyle_df["ligand"].nunique() == 4
    assert doyle_df["halide"].nunique() == 15
    assert doyle_df["additive"].nunique() == 24  # 23 + NONE


def test_doyle_yield_range(doyle_df: pd.DataFrame) -> None:
    y = doyle_df["yield"]
    assert y.min() >= 0.0
    assert y.max() <= 100.0


def test_doyle_full_joint_is_all_singletons(doyle_df: pd.DataFrame) -> None:
    """Under the full (base, ligand, halide, additive) descriptor, every
    row is unique -- the gate-1 signature of a no-replicate full factorial."""
    key_counts = (
        doyle_df[["base", "ligand", "halide", "additive"]]
        .value_counts()
    )
    assert key_counts.max() == 1
    assert key_counts.min() == 1


def test_doyle_columns_present(doyle_df: pd.DataFrame) -> None:
    """Rich DataFrame: all expected ancillary columns present."""
    expected = {
        "plate", "row", "col",
        "base", "base_cas_number", "base_smiles",
        "ligand", "ligand_cas_number", "ligand_smiles",
        "halide", "halide_smiles",
        "additive", "additive_smiles",
        "product_smiles", "yield",
    }
    assert expected.issubset(doyle_df.columns)
