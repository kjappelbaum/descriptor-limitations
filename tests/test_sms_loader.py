"""Tests for the SMS loader."""

from __future__ import annotations

import pandas as pd
import pytest

from descriptor_limitations.data_loaders import load_sms


@pytest.fixture(scope="module")
def sms_df() -> pd.DataFrame:
    return load_sms()


def test_sms_shape(sms_df: pd.DataFrame) -> None:
    assert len(sms_df) == 484


def test_sms_no_spacer_columns(sms_df: pd.DataFrame) -> None:
    for c in sms_df.columns:
        assert not str(c).startswith("Unnamed:")


def test_sms_categories(sms_df: pd.DataFrame) -> None:
    """Four categories: optimization, chiralty (sic), review, generated."""
    counts = sms_df["Category"].value_counts().to_dict()
    assert counts == {
        "optimization": 299, "chiralty": 136, "review": 30, "generated": 19
    }


def test_sms_is_generated_flag(sms_df: pd.DataFrame) -> None:
    assert sms_df["is_generated"].sum() == 19
    # All generated rows have outcome 'incompatible'.
    assert (sms_df.loc[sms_df["is_generated"], "outcome"] == "incompatible").all()


def test_sms_score_range(sms_df: pd.DataFrame) -> None:
    assert sms_df["score"].min() >= 0.0
    assert sms_df["score"].max() <= 1.0


def test_sms_binary_success_outcome(sms_df: pd.DataFrame) -> None:
    """successful + good = 316."""
    assert sms_df["binary_success_outcome"].sum() == 316


def test_sms_binary_success_score(sms_df: pd.DataFrame) -> None:
    """score >= 0.5 = 351."""
    assert sms_df["binary_success_score"].sum() == 351


def test_sms_required_columns(sms_df: pd.DataFrame) -> None:
    for col in [
        "ligand name", "T [°C]", "t [h]", "solvent1", "metal",
        "inorganic salt", "outcome", "score", "Category",
        "is_generated", "binary_success_outcome", "binary_success_score",
    ]:
        assert col in sms_df.columns
