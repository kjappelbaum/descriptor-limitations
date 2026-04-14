"""Dataset loaders.

Each loader downloads raw data on first call, caches it under
``data/<dataset>/raw/``, and returns a tidy pandas DataFrame. Raw files
are gitignored; this module is the single source of truth for how
datasets enter the analysis pipeline.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

# Repo root: <repo>/src/descriptor_limitations/data_loaders.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"


# -- Doyle Buchwald-Hartwig -------------------------------------------------

# Source: Ahneman, Estrada, Lin, Dreher, Doyle. "Predicting reaction
# performance in C-N cross-coupling using machine learning." Science 360,
# 186-190 (2018). doi:10.1126/science.aar5169
# Raw CSV from the authors' reference implementation:
#   https://github.com/doylelab/rxnpredict
_DOYLE_URL = (
    "https://raw.githubusercontent.com/doylelab/rxnpredict/master/data_table.csv"
)
_DOYLE_RAW = _DATA_DIR / "doyle" / "raw" / "data_table.csv"

# Expected factor levels after cleaning (see docstring of `load_doyle`).
_DOYLE_EXPECTED_LEVELS = {
    "base": 3,
    "ligand": 4,
    "halide": 15,
    "additive": 24,  # 23 named additives + 'NONE' (no-additive control)
}


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def load_doyle(*, cache_dir: Path | None = None) -> pd.DataFrame:
    """Load the Doyle Buchwald-Hartwig HTE dataset.

    Downloads ``data_table.csv`` from the Doyle lab's ``rxnpredict`` repo
    on first call (cached under ``data/doyle/raw/``). Returns a tidy
    DataFrame with the four categorical reaction factors, the continuous
    yield, and all ancillary columns (SMILES, CAS numbers, plate
    coordinates) preserved for downstream analysis.

    Cleaning
    --------
    * The raw CSV has 4599 rows.
    * 287 rows with missing ``aryl_halide`` (and missing ``product_smiles``)
      are blank wells with no reaction possible; these are dropped.
    * 192 rows with missing ``additive`` are the "no additive" control
      condition (yields 0-100%, mean 46%). These are kept; ``additive``
      is recoded to the string ``'NONE'`` so the control is a
      first-class factor level.
    * ``aryl_halide`` is renamed to ``halide`` throughout.

    After cleaning, the returned DataFrame has 4599 - 287 = 4312 rows,
    with 3 bases x 4 ligands x 15 halides x 24 additives (23 named + NONE)
    as factor levels. Not every level combination is populated; under the
    full joint descriptor, every row is a singleton (max duplicates = 1).
    This is the expected signature of a fixed full-factorial design
    without replicates and is why `r2_ceiling` at full resolution returns
    the trivial bracket [0, 1] -- gate 1 in CLAUDE.md.

    Parameters
    ----------
    cache_dir : Path, optional
        Override the default cache location (``<repo>/data/doyle/raw``).
        Mainly for tests.

    Returns
    -------
    df : pd.DataFrame
        Columns (at minimum): plate, row, col, base, base_cas_number,
        base_smiles, ligand, ligand_cas_number, ligand_smiles,
        halide, halide_smiles, additive, additive_smiles,
        product_smiles, yield.

    Raises
    ------
    RuntimeError
        If the post-cleaning DataFrame does not have the expected factor
        cardinalities. A mismatch indicates the upstream CSV has changed
        and loader assumptions need revisiting.
    """
    raw_path = (
        _DOYLE_RAW if cache_dir is None else Path(cache_dir) / "data_table.csv"
    )
    _download_if_missing(_DOYLE_URL, raw_path)

    df = pd.read_csv(raw_path)

    # Drop wells with no aryl halide (blank wells, no reaction).
    df = df[df["aryl_halide"].notna()].copy()

    # Rename to project convention.
    df = df.rename(
        columns={
            "aryl_halide": "halide",
            "aryl_halide_number": "halide_number",
            "aryl_halide_smiles": "halide_smiles",
        }
    )

    # Recode missing additive to the explicit 'NONE' factor level.
    df["additive"] = df["additive"].fillna("NONE")

    # Validate expected cardinalities so upstream drift is caught loudly.
    for col, expected in _DOYLE_EXPECTED_LEVELS.items():
        got = df[col].nunique()
        if got != expected:
            raise RuntimeError(
                f"load_doyle: {col} has {got} unique levels, expected "
                f"{expected}. Upstream data may have changed."
            )

    # Yield sanity check: percentage in [0, 100].
    y = df["yield"]
    if not ((y >= 0) & (y <= 100)).all():
        raise RuntimeError(
            f"load_doyle: yield out of [0, 100] range "
            f"(min={y.min()}, max={y.max()})."
        )

    return df.reset_index(drop=True)
