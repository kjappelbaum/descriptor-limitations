"""Case Study 5: Aqueous solubility (AqSolDB) -- within-SMILES ceiling.

AqSolDB merges 9 public solubility datasets and preserves per-source
LogS measurements (19795 rows, 9982 unique InChIKeys, 4044 compounds
with multi-source replicates). This notebook computes the descriptor
ceiling from within-SMILES variance and compares to published solubility
models. In contrast to polymer Tg, the published models are *below*
the ceiling -- indicating algorithmic headroom on this task, not
descriptor insufficiency.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from descriptor_limitations.data_loaders import load_aqsoldb_sources
    from descriptor_limitations.information import r2_ceiling

    # Published points of reference (all LogS units, mol/L):
    PUBLISHED_DMPNN_RMSE = 0.555   # Yang et al. 2019 (ESOL benchmark)
    PUBLISHED_AQSOLPRED_RMSE = 0.77  # Sorkun et al. 2021 (AqSolDB random split)
    return (
        PUBLISHED_AQSOLPRED_RMSE,
        PUBLISHED_DMPNN_RMSE,
        load_aqsoldb_sources,
        mo,
        np,
        pd,
        r2_ceiling,
    )


@app.cell
def _(load_aqsoldb_sources, pd):
    srcs = load_aqsoldb_sources()
    per_ik = srcs.groupby("InChIKey").size()
    print(f"source rows:          {len(srcs)}")
    print(f"unique InChIKeys:     {per_ik.size}")
    print(f"InChIKeys with >=2:   {(per_ik>=2).sum()}")
    print(f"InChIKeys with >=3:   {(per_ik>=3).sum()}")
    print()
    print(f"LogS (mol/L) summary over all source rows:")
    print(srcs["Solubility"].describe().round(3).to_string())
    return per_ik, srcs


@app.cell
def _(np, pd, r2_ceiling, srcs, per_ik):
    """Ceiling ladder on three subsets. All-measurements uses InChIKey
    singletons and thus has a wide bracket; multi-source subsets have
    no singletons, so the bracket collapses."""

    def _row(subset_df, label):
        y = subset_df["Solubility"].to_numpy()
        g = subset_df["InChIKey"].to_numpy()
        r = r2_ceiling(y, g)
        wgv_opt = (1.0 - r.optimistic) * r.var_y
        wgv_pes = (1.0 - r.pessimistic) * r.var_y
        mae_opt = float(np.sqrt(2.0 * wgv_opt / np.pi))
        mae_pes = float(np.sqrt(2.0 * wgv_pes / np.pi))
        rmse_opt = float(np.sqrt(wgv_opt))
        rmse_pes = float(np.sqrt(wgv_pes))
        return {
            "subset": label,
            "n_measurements": len(subset_df),
            "n_compounds": r.n_groups,
            "n_singletons": r.n_singletons,
            "R2_pess": round(r.pessimistic, 4),
            "R2_opt": round(r.optimistic, 4),
            "MAE_floor_opt_LogS": round(mae_opt, 3),
            "MAE_floor_pess_LogS": round(mae_pes, 3),
            "RMSE_floor_opt_LogS": round(rmse_opt, 3),
            "RMSE_floor_pess_LogS": round(rmse_pes, 3),
        }

    two_plus = srcs[srcs["InChIKey"].isin(per_ik[per_ik >= 2].index)]
    three_plus = srcs[srcs["InChIKey"].isin(per_ik[per_ik >= 3].index)]

    rows = [
        _row(srcs, "all measurements (incl. 5938 single-source compounds)"),
        _row(two_plus, "multi-source (>=2 measurements)"),
        _row(three_plus, "well-replicated (>=3 measurements)"),
    ]
    ceiling_table = pd.DataFrame(rows)
    ceiling_table
    return ceiling_table, three_plus, two_plus


@app.cell
def _(np, pd, three_plus, two_plus):
    """Per-compound within-source SD distribution (LogS units).
    The median and tail tell us where most of the 'irreducible' noise
    lives."""
    def _sd_summary(df, label):
        per = df.groupby("InChIKey")["Solubility"].std(ddof=0)
        return {
            "subset": label,
            "n_compounds": per.size,
            "SD_mean": round(per.mean(), 3),
            "SD_median": round(per.median(), 3),
            "SD_p90": round(np.quantile(per, 0.90), 3),
            "SD_p99": round(np.quantile(per, 0.99), 3),
            "SD_max": round(per.max(), 3),
        }
    sd_table = pd.DataFrame([
        _sd_summary(two_plus, ">=2 sources"),
        _sd_summary(three_plus, ">=3 sources"),
    ])
    sd_table
    return (sd_table,)


@app.cell
def _(
    PUBLISHED_AQSOLPRED_RMSE,
    PUBLISHED_DMPNN_RMSE,
    ceiling_table,
    mo,
    sd_table,
):
    mo.md(
        f"""
        ## Descriptor ceiling on AqSolDB

        ```
{ceiling_table.to_string(index=False)}
        ```

        ## Within-SMILES SD (LogS) distribution

        ```
{sd_table.to_string(index=False)}
        ```

        ## Published reference points

        | Model | Split / dataset | RMSE (LogS) |
        |---|---|---|
        | D-MPNN (Yang et al. 2019) | MoleculeNet ESOL scaffold split | {PUBLISHED_DMPNN_RMSE} |
        | AqSolPred (Sorkun 2021)   | AqSolDB random split | {PUBLISHED_AQSOLPRED_RMSE} |

        ## Headline reading

        On the multi-source subset (n=13857, 4044 compounds, zero
        singletons), the within-SMILES descriptor ceiling is
        **R^2 = 0.971** with an **RMSE floor of 0.40 LogS** (equivalent
        MAE floor 0.32 LogS under the half-normal residual assumption).

        Published solubility SOTA sits **above** this floor:

        * D-MPNN: RMSE 0.555 LogS on ESOL (a smaller, cleaner subset
          of AqSolDB). Compare to ceiling RMSE 0.40 LogS on the full
          multi-source AqSolDB -- but note the population mismatch;
          ESOL may have lower within-SMILES noise than the full set.
        * AqSolPred: RMSE 0.77 LogS on an AqSolDB random split.
          Above the ceiling by a factor of ~1.9.

        **The gap between current models and the ceiling is real
        algorithmic headroom -- room for better architectures or
        training protocols to close on AqSolDB-scale data without
        needing new descriptors.** This contrasts sharply with polymer
        Tg (Case Study 2), where the best models already sit at the
        floor and further improvement requires descriptors beyond
        PSMILES.

        ## What this case study adds to the paper

        * A canonical ML-for-chemistry task (aqueous solubility on
          AqSolDB / ESOL) with a concrete, defensible ceiling.
        * The first case in the paper where the framework's
          prescription is **"invest in algorithms, not descriptors"**
          -- the complement of the polymer Tg prescription. Together
          the two cases show that the framework yields *actionable,
          task-specific* guidance rather than a blanket "ML has
          limits" claim.
        * A cross-check on the polymer Tg methodology: AqSolDB gives
          a larger replicate corpus (4044 multi-source compounds) at
          scale, confirming that the within-identity variance
          estimator behaves well on real data.

        ## Caveats

        * The comparison between AqSolDB ceiling and ESOL RMSE is
          across populations: ESOL is ~1128 molecules, a curated
          subset. A fair comparison retrains D-MPNN on AqSolDB or
          recomputes the ceiling on ESOL-only compounds. Current
          numbers are indicative, not apples-to-apples.
        * AqSolDB aggregates experimental solubility from 9 sources,
          spanning different measurement methods (shake-flask,
          CheqSol, computational estimates sometimes mixed in).
          Part of the ceiling SD reflects method heterogeneity,
          not irreducible molecular variance. Filtering by method
          where reported would tighten the ceiling further.
        * Half-normal residual assumption inflates MAE relative to
          the RMSE floor if actual residuals are sub-Gaussian. R^2
          and RMSE floor are distribution-free.
        """
    )
    return


if __name__ == "__main__":
    app.run()
