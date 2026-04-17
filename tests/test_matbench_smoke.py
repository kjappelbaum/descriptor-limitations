"""Smoke tests for matbench loaders and notebook."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from descriptor_limitations.data_loaders import (
    load_matbench_expt_gap,
    load_matbench_mp_gap_compositions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "matbench_gap_ceiling.py"


@pytest.fixture(scope="module")
def expt() -> pd.DataFrame:
    return load_matbench_expt_gap()


@pytest.fixture(scope="module")
def mp() -> pd.DataFrame:
    return load_matbench_mp_gap_compositions()


def test_expt_gap_shape(expt: pd.DataFrame) -> None:
    assert len(expt) == 4604


def test_mp_gap_shape(mp: pd.DataFrame) -> None:
    assert len(mp) == 106113
    assert mp["reduced_formula"].nunique() == 78164


def test_matbench_notebook_runs(tmp_path: Path) -> None:
    script = tmp_path / "mb_run.py"
    subprocess.run(
        ["marimo", "export", "script", str(NOTEBOOK), "-o", str(script)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    res = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert res.returncode == 0, (
        f"notebook failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
