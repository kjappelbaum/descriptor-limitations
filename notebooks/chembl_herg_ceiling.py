"""Case Study 7: ChEMBL hERG IC50 -- mixed regime (headroom + ceiling).

hERG potassium channel inhibition is a key safety endpoint in drug
discovery. Same compound tested across different assays, cell lines,
and protocols gives different IC50 values -- this is regime-1 variance
(physically different conditions). The within-compound variance sets a
descriptor ceiling meaningfully below 1.0, while published QSAR SOTA
is also well below the ceiling. The result is a three-way variance
decomposition: explained, algorithmic headroom, and irreducible.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    from descriptor_limitations.information import r2_ceiling

    PUBLISHED_HERG_R2 = 0.59       # general hERG IC50 QSAR
    PUBLISHED_HERG_RMSE = 0.73     # pIC50 units
    PUBLISHED_HERGBOOST_R2 = 0.394 # hERGBoost 2024, external val
    PUBLISHED_HERGBOOST_RMSE = 0.616
    return (
        Path,
        PUBLISHED_HERG_R2,
        PUBLISHED_HERG_RMSE,
        PUBLISHED_HERGBOOST_R2,
        PUBLISHED_HERGBOOST_RMSE,
        json,
        mo,
        np,
        pd,
        r2_ceiling,
    )


@app.cell
def _(Path, json, np, pd):
    """Load hERG IC50 data from ChEMBL (pre-fetched via REST API)."""
    # When running as an exported script, __file__ may resolve
    # differently; use a fallback path search.
    _candidates = [
        Path(__file__).resolve().parents[1] / "data" / "chembl" / "raw" / "chembl_herg_ic50.json",
        Path.cwd() / "data" / "chembl" / "raw" / "chembl_herg_ic50.json",
    ]
    cache = next((p for p in _candidates if p.exists()), _candidates[-1])
    with open(cache) as f:
        acts = json.load(f)
    rows = []
    for a in acts:
        sv, smi, pv, mid = (a.get("standard_value"), a.get("canonical_smiles"),
                             a.get("pchembl_value"), a.get("molecule_chembl_id"))
        if sv and smi and pv:
            try:
                rows.append({
                    "mol_id": mid, "smiles": smi,
                    "ic50_nM": float(sv), "pIC50": float(pv),
                    "assay_id": a.get("assay_chembl_id"),
                })
            except (ValueError, TypeError):
                pass
    herg = pd.DataFrame(rows)
    per_mol = herg.groupby("mol_id").size()
    print(f"Valid hERG IC50 records: {len(herg)}")
    print(f"Unique molecules: {per_mol.size}")
    print(f"Unique assays: {herg['assay_id'].nunique()}")
    print(f"Molecules with >=2 measurements: {(per_mol>=2).sum()}")
    print(f"Molecules with >=3 measurements: {(per_mol>=3).sum()}")
    print(f"\npIC50 range: {herg['pIC50'].min():.2f} – {herg['pIC50'].max():.2f}")
    return herg, per_mol


@app.cell
def _(herg, np, pd, per_mol, r2_ceiling):
    """Ceiling from within-compound IC50 variance across assays."""
    multi_ids = per_mol[per_mol >= 2].index
    multi_herg = herg[herg["mol_id"].isin(multi_ids)]
    three_ids = per_mol[per_mol >= 3].index
    three_herg = herg[herg["mol_id"].isin(three_ids)]

    def _row(df, label):
        r = r2_ceiling(df["pIC50"].to_numpy(), df["mol_id"].to_numpy())
        wgv = (1 - r.optimistic) * r.var_y
        return {
            "subset": label,
            "n_measurements": len(df),
            "n_compounds": r.n_groups,
            "R2_ceiling": round(r.optimistic, 4),
            "RMSE_floor": round(float(np.sqrt(wgv)), 3),
            "MAE_floor": round(float(np.sqrt(2 * wgv / np.pi)), 3),
        }

    ceil_table = pd.DataFrame([
        _row(multi_herg, ">=2 assay measurements"),
        _row(three_herg, ">=3 assay measurements"),
    ])

    sd_per = multi_herg.groupby("mol_id")["pIC50"].std(ddof=0)
    sd_stats = pd.DataFrame([{
        "stat": "within-compound pIC50 SD",
        "mean": round(sd_per.mean(), 3),
        "median": round(sd_per.median(), 3),
        "p90": round(float(np.quantile(sd_per, 0.90)), 3),
        "max": round(sd_per.max(), 3),
    }])
    return ceil_table, sd_stats


@app.cell
def _(
    PUBLISHED_HERG_R2,
    PUBLISHED_HERG_RMSE,
    PUBLISHED_HERGBOOST_R2,
    PUBLISHED_HERGBOOST_RMSE,
    ceil_table,
    mo,
    sd_stats,
):
    ceiling_r2 = ceil_table.iloc[0]["R2_ceiling"]
    irreducible = round(1.0 - ceiling_r2, 3)
    headroom = round(ceiling_r2 - PUBLISHED_HERG_R2, 3)
    explained = round(PUBLISHED_HERG_R2, 3)

    mo.md(
        f"""
        ## hERG IC50 descriptor ceiling from multi-assay variance

        ```
{ceil_table.to_string(index=False)}
        ```

        Within-compound pIC50 SD distribution:
        ```
{sd_stats.to_string(index=False)}
        ```

        ## Published reference points

        | Model | R² | RMSE (pIC50) | Source |
        |---|---|---|---|
        | General hERG IC50 QSAR | {PUBLISHED_HERG_R2} | {PUBLISHED_HERG_RMSE} | Literature consensus |
        | hERGBoost (2024, ext. val.) | {PUBLISHED_HERGBOOST_R2} | {PUBLISHED_HERGBOOST_RMSE} | Cai et al. 2024 |

        ## Three-way variance decomposition

        | Component | R² share | Meaning |
        |---|---|---|
        | Explained by SOTA | **{explained}** | What current models capture |
        | Algorithmic headroom | **{headroom}** | Room for better models within SMILES |
        | Irreducible (assay variability) | **{irreducible}** | Same compound, different assay → different IC50 |

        ## Why this is regime (1), not regime (3)

        The within-compound IC50 variance comes from the compound being
        tested in **genuinely different biological assays**: different
        cell lines, different expression systems, different
        concentrations, different electrophysiology protocols. This is
        NOT measurement noise on the same experiment — it is physically
        meaningful variation that SMILES cannot encode because SMILES
        does not know what assay the compound will be tested in.

        A model that predicts "pIC50 of compound X against hERG" is
        implicitly predicting a **distribution** of outcomes, not a
        single number. The ceiling says: the best any SMILES-only
        model can do is predict the mean of this distribution, leaving
        {irreducible} of variance unexplained.

        ## What this case study adds to the paper

        * Extends the framework to **drug discovery** — the largest
          ML-for-chemistry application.
        * First case with a **mixed regime**: neither fully saturated
          (R²=0.86 > SOTA=0.59) nor fully headroom (ceiling < 1.0).
          The three-way decomposition is a novel and actionable output.
        * Uses genuinely different physical conditions (assays) as
          replicates — addresses the methodological point that multi-
          source variance is only informative when it captures real
          variation, not just measurement noise.
        * Quantifies the classic QSAR claim "experimental uncertainty
          limits model accuracy" — yes, but only 14% of variance;
          the other 27% is algorithmic headroom that better models
          can close.
        """
    )
    return


if __name__ == "__main__":
    app.run()
