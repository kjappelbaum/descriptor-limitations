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

    # Full threshold sensitivity table
    thresh_rows = []
    for min_m in [2, 3, 5, 10]:
        ids_t = per_mol[per_mol >= min_m].index
        sub_t = herg[herg["mol_id"].isin(ids_t)]
        if sub_t["mol_id"].nunique() < 10:
            continue
        r_t = r2_ceiling(sub_t["pIC50"].to_numpy(), sub_t["mol_id"].to_numpy())
        wgv_t = (1 - r_t.optimistic) * r_t.var_y
        thresh_rows.append({
            "min_measurements": min_m,
            "n_measurements": len(sub_t),
            "n_compounds": r_t.n_groups,
            "R2_ceiling": round(r_t.optimistic, 4),
            "RMSE_floor": round(float(np.sqrt(wgv_t)), 3),
        })
    thresh_table = pd.DataFrame(thresh_rows)

    # Fano binary analysis (blocker = pIC50 > 5, i.e. IC50 < 10 µM)
    from descriptor_limitations.information import conditional_entropy, fano_bound, predictability
    multi_herg_copy = multi_herg.copy()
    multi_herg_copy["blocker"] = (multi_herg_copy["pIC50"] > 5).astype(int)
    y_bin = multi_herg_copy["blocker"].to_numpy()
    X_bin = multi_herg_copy["mol_id"].to_numpy()
    H_cond_herg = conditional_entropy(y_bin, X_bin, correction="miller-madow")
    pe_herg = fano_bound(H_cond_herg, 2, variant="tight")
    pi_herg = predictability(H_cond_herg, 2)
    base_err_herg = float(min(y_bin.mean(), 1 - y_bin.mean()))

    fano_result = pd.DataFrame([{
        "threshold": "pIC50 > 5 (IC50 < 10 µM)",
        "blocker_rate": round(float(y_bin.mean()), 3),
        "H(blocker|mol_id)_bits": round(H_cond_herg, 3),
        "Fano_Pe_min": round(pe_herg, 3),
        "predictability": round(pi_herg, 3),
        "base_rate_error": round(base_err_herg, 3),
    }])

    sd_per = multi_herg.groupby("mol_id")["pIC50"].std(ddof=0)
    sd_stats = pd.DataFrame([{
        "stat": "within-compound pIC50 SD",
        "mean": round(sd_per.mean(), 3),
        "median": round(sd_per.median(), 3),
        "p90": round(float(np.quantile(sd_per, 0.90)), 3),
        "max": round(sd_per.max(), 3),
    }])
    return fano_result, sd_stats, thresh_table


@app.cell
def _(
    PUBLISHED_HERG_R2,
    PUBLISHED_HERG_RMSE,
    PUBLISHED_HERGBOOST_R2,
    PUBLISHED_HERGBOOST_RMSE,
    fano_result,
    mo,
    sd_stats,
    thresh_table,
):
    ceiling_r2 = thresh_table.iloc[0]["R2_ceiling"]
    irreducible = round(1.0 - ceiling_r2, 3)
    headroom = round(ceiling_r2 - PUBLISHED_HERG_R2, 3)
    explained = round(PUBLISHED_HERG_R2, 3)

    mo.md(
        f"""
        ## hERG IC50 descriptor ceiling — threshold sensitivity

        The ceiling depends on assay diversity: compounds tested in
        more assays show more within-compound variance (lower ceiling).

        ```
{thresh_table.to_string(index=False)}
        ```

        Within-compound pIC50 SD distribution (>=2 measurements):
        ```
{sd_stats.to_string(index=False)}
        ```

        ## Fano binary analysis (hERG blocker classification)

        ```
{fano_result.to_string(index=False)}
        ```

        At the binary level (blocker vs non-blocker at IC50 = 10 µM),
        compound identity nearly determines blocker status:
        H(blocker|mol_id) = 0.148 bits, Fano P_e >= 2.1%. The binary
        classification ceiling is very high (~98% accuracy achievable).
        The challenge in hERG prediction is not binary classification
        but **continuous IC50 ranking** — which is where the R^2
        ceiling of 0.65-0.86 applies.

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
