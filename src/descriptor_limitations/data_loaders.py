"""Dataset loaders.

Each loader downloads raw data on first call, caches it under
``data/<dataset>/raw/``, and returns a tidy pandas DataFrame. Raw files
are gitignored; this module is the single source of truth for how
datasets enter the analysis pipeline.
"""

from __future__ import annotations

import ast
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


# -- PolyMetriX curated polymer Tg dataset ---------------------------------

# Source: Kunchapu & Jablonka. "PolyMetriX: an ecosystem for digital polymer
# chemistry." npj Comput. Mater. 11, 312 (2025). doi:10.1038/s41524-025-01823-y
# Curated dataset on Zenodo: doi:10.5281/zenodo.15210035
_POLYTG_URL = (
    "https://zenodo.org/records/15210035/files/"
    "LAMALAB_CURATED_Tg_structured_polymerclass.csv?download=1"
)
_POLYTG_RAW = (
    _DATA_DIR / "polymer_tg" / "raw"
    / "LAMALAB_CURATED_Tg_structured_polymerclass.csv"
)


def load_polymetrix_tg(*, cache_dir: Path | None = None) -> pd.DataFrame:
    """Load the PolyMetriX curated polymer Tg dataset.

    One row per unique PSMILES (7367 rows). The full set of source-level
    Tg measurements is preserved in ``meta.tg_values`` (a string-encoded
    Python list). Use `expand_polymetrix_tg` to expand to long form
    (one row per individual measurement) for ceiling analysis.

    The dataset has four reliability tiers:

    * black  (7088 rows) -- single source per PSMILES (singletons)
    * yellow ( 132 rows) -- two sources, agreement (Z <= 2)
    * gold   ( 143 rows) -- three or more sources, agreement (Z <= 2)
    * red    (   4 rows) -- multiple sources, disagreement (Z > 2)

    Yellow+Gold+Red rows carry within-PSMILES Tg variance and are the
    informative subset for r2_ceiling computation.

    Parameters
    ----------
    cache_dir : Path, optional
        Override the default cache location (``data/polymer_tg/raw``).

    Returns
    -------
    df : pd.DataFrame
        112-column curated CSV as published, with columns including
        ``PSMILES``, ``labels.Exp_Tg(K)``, ``meta.tg_values``,
        ``meta.num_of_points``, ``meta.std``, ``meta.reliability``,
        and PolyMetriX features.
    """
    raw_path = (
        _POLYTG_RAW
        if cache_dir is None
        else Path(cache_dir) / "LAMALAB_CURATED_Tg_structured_polymerclass.csv"
    )
    _download_if_missing(_POLYTG_URL, raw_path)
    df = pd.read_csv(raw_path)
    if "PSMILES" not in df.columns or "labels.Exp_Tg(K)" not in df.columns:
        raise RuntimeError(
            "load_polymetrix_tg: expected columns missing -- upstream CSV "
            "schema may have changed."
        )
    return df


def expand_polymetrix_tg(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the curated Tg dataset to one row per individual measurement.

    For singletons (``num_of_points == 1`` and ``meta.tg_values`` empty),
    the row's ``labels.Exp_Tg(K)`` is the single measurement. For multi-
    source rows, ``meta.tg_values`` holds a list literal of all reported
    Tg values across sources; each is emitted as a separate row.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `load_polymetrix_tg`.

    Returns
    -------
    long : pd.DataFrame
        Columns: ``psmiles``, ``tg_K``, ``reliability``,
        ``num_of_points``, ``measurement_index`` (0-based within group).

    Raises
    ------
    ValueError
        If a multi-source row's ``meta.tg_values`` cannot be parsed or
        its length disagrees with ``meta.num_of_points``.
    """
    rows = []
    for _, r in df.iterrows():
        psmiles = r["PSMILES"]
        n = int(r["meta.num_of_points"])
        reliability = r["meta.reliability"]
        tg_values_raw = r["meta.tg_values"]
        if n == 1:
            measurements = [float(r["labels.Exp_Tg(K)"])]
        else:
            try:
                measurements = ast.literal_eval(tg_values_raw)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"Cannot parse meta.tg_values for PSMILES={psmiles}: "
                    f"{tg_values_raw!r} ({e})"
                )
            if len(measurements) != n:
                raise ValueError(
                    f"meta.tg_values length {len(measurements)} disagrees "
                    f"with meta.num_of_points {n} for PSMILES={psmiles}"
                )
        for i, tg in enumerate(measurements):
            rows.append({
                "psmiles": psmiles,
                "tg_K": float(tg),
                "reliability": reliability,
                "num_of_points": n,
                "measurement_index": i,
            })
    return pd.DataFrame(rows)


# -- SMS: Solvothermal MOF Synthesis ---------------------------------------

# Source: Novotny et al. "SMS: A curated dataset on solvothermal MOF
# syntheses." (Zenodo 15045511). Raw CSV from the authors' reference repo:
#   https://github.com/JorenBE/SMS/tree/main/experiments/Supervised
_SMS_URL = (
    "https://raw.githubusercontent.com/JorenBE/SMS/main/"
    "experiments/Supervised/20240402_syntheses_UTF8.csv"
)
_SMS_RAW = _DATA_DIR / "sms" / "raw" / "20240402_syntheses_UTF8.csv"

# Outcomes counted as "success" under the primary binary target.
_SMS_SUCCESS_OUTCOMES = frozenset({"successful", "good"})


def load_sms(*, cache_dir: Path | None = None) -> pd.DataFrame:
    """Load the SMS solvothermal MOF synthesis dataset.

    Returns 484 curated MOF syntheses with mixed categorical and
    continuous descriptors, a continuous ``score`` (0.00-1.00), and a
    17-level ``outcome`` label. 19 of the 484 entries are expert-
    generated infeasibility controls (``Category == 'generated'``,
    ``outcome == 'incompatible'``) rather than literature-reported
    experiments; these are flagged via the boolean ``is_generated``
    column so downstream code can report ceilings with and without them.

    Two binary success targets are added for convenience:

    * ``binary_success_outcome``: ``outcome in {'successful', 'good'}``
      (316 / 484 = 65%).
    * ``binary_success_score``: ``score >= 0.5``
      (351 / 484 = 73%).

    The original two unlabeled spacer columns (``Unnamed: 15`` and
    ``Unnamed: 20``) are dropped.

    Parameters
    ----------
    cache_dir : Path, optional
        Override the default cache location (``data/sms/raw``).

    Returns
    -------
    df : pd.DataFrame
        Columns include: ligand name, T [C], t [h], solvent1-3 with
        volume fractions, inorganic salt, metal, additional, reported
        DOI, outcome, score, CCDC, Category, plus the derived
        ``is_generated``, ``binary_success_outcome``,
        ``binary_success_score``.
    """
    raw_path = (
        _SMS_RAW
        if cache_dir is None
        else Path(cache_dir) / "20240402_syntheses_UTF8.csv"
    )
    _download_if_missing(_SMS_URL, raw_path)
    df = pd.read_csv(raw_path)

    # Drop spacer columns if present.
    df = df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed:")],
        errors="ignore",
    )

    if len(df) != 484:
        raise RuntimeError(
            f"load_sms: expected 484 rows, got {len(df)}. Upstream data "
            "may have changed."
        )

    expected_categories = {"optimization", "chiralty", "review", "generated"}
    got_categories = set(df["Category"].unique())
    if got_categories != expected_categories:
        raise RuntimeError(
            f"load_sms: Category set {got_categories} differs from expected "
            f"{expected_categories}."
        )

    # Score range check.
    if not ((df["score"] >= 0.0) & (df["score"] <= 1.0)).all():
        raise RuntimeError(
            f"load_sms: score out of [0, 1] range "
            f"(min={df['score'].min()}, max={df['score'].max()})."
        )

    # Derived flags.
    df["is_generated"] = df["Category"] == "generated"
    df["binary_success_outcome"] = df["outcome"].isin(_SMS_SUCCESS_OUTCOMES)
    df["binary_success_score"] = df["score"] >= 0.5

    return df.reset_index(drop=True)
