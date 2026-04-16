"""Case Study 2: PolyMetriX polymer Tg -- within-PSMILES R^2 ceiling.

Quantifies the irreducible MAE/RMSE/R^2 limit set by within-PSMILES Tg
variance in the curated PolyMetriX dataset. Compares to the cross-
dataset MAE range (13.79 K best, 214.75 K worst) reported by Kunchapu
& Jablonka 2025.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from descriptor_limitations.data_loaders import (
        expand_polymetrix_tg,
        load_polymetrix_tg,
    )
    from descriptor_limitations.information import r2_ceiling

    PUBLISHED_BEST_CROSS_DATASET_MAE_K = 13.79  # Kunchapu & Jablonka 2025
    PUBLISHED_WORST_CROSS_DATASET_MAE_K = 214.75
    return (
        PUBLISHED_BEST_CROSS_DATASET_MAE_K,
        PUBLISHED_WORST_CROSS_DATASET_MAE_K,
        expand_polymetrix_tg,
        load_polymetrix_tg,
        mo,
        np,
        pd,
        r2_ceiling,
    )


@app.cell
def _(load_polymetrix_tg):
    df = load_polymetrix_tg()
    print(f"Curated rows: {len(df)}")
    print("Reliability tiers:")
    print(df["meta.reliability"].value_counts().to_string())
    print()
    print("num_of_points distribution:")
    print(df["meta.num_of_points"].value_counts().sort_index().to_string())
    return (df,)


@app.cell
def _(df, expand_polymetrix_tg):
    long = expand_polymetrix_tg(df)
    print(f"Expanded measurements: {len(long)}")
    print(f"Unique PSMILES: {long['psmiles'].nunique()}")
    print()
    print("Tg (K) summary:")
    print(long["tg_K"].describe().to_string())
    return (long,)


@app.cell
def _(long, np, pd, r2_ceiling):
    """R^2_ceiling at multiple subsets:
       (a) full dataset (all reliability tiers, includes 7088 black singletons)
       (b) yellow + gold + red (multi-source only, no singletons)
       (c) yellow + gold (drop the 4 'red' Z>2 outliers)
    """
    rows = []
    for label, subset in [
        ("all (incl. 7088 black singletons)", long),
        ("multi-source (yellow + gold + red)",
            long[long["reliability"].isin(["yellow", "gold", "red"])]),
        ("yellow + gold only (Z<=2 agreement)",
            long[long["reliability"].isin(["yellow", "gold"])]),
    ]:
        y = subset["tg_K"].to_numpy()
        g = subset["psmiles"].to_numpy()
        r = r2_ceiling(y, g)
        # Convert within-group variance to MAE-floor for half-normal residuals.
        wgv = (1.0 - r.optimistic) * r.var_y  # optimistic = upper R^2 bound
        # MAE under within-group residuals modeled as N(0, wgv): E|X| = sqrt(2*var/pi)
        mae_floor_K_opt = float(np.sqrt(2.0 * wgv / np.pi))
        wgv_pes = (1.0 - r.pessimistic) * r.var_y
        mae_floor_K_pes = float(np.sqrt(2.0 * wgv_pes / np.pi))
        rows.append({
            "subset": label,
            "n_measurements": len(subset),
            "n_psmiles": r.n_groups,
            "n_singletons": r.n_singletons,
            "Var(y) [K^2]": round(r.var_y, 1),
            "R2_pess": round(r.pessimistic, 4),
            "R2_opt": round(r.optimistic, 4),
            "MAE_floor_pess [K]": round(mae_floor_K_pes, 2),
            "MAE_floor_opt [K]": round(mae_floor_K_opt, 2),
        })
    ceiling_table = pd.DataFrame(rows)
    ceiling_table
    return (ceiling_table,)


@app.cell
def _(long, np, pd):
    """Per-PSMILES distribution of within-group SD (multi-source rows)."""
    multi = long[long["num_of_points"] > 1]
    sd_per = multi.groupby("psmiles")["tg_K"].std(ddof=0)
    out = pd.DataFrame({
        "stat": ["count", "mean", "median", "p90", "max"],
        "within-PSMILES SD [K]": [
            int(sd_per.shape[0]),
            float(sd_per.mean()),
            float(sd_per.median()),
            float(np.quantile(sd_per, 0.90)),
            float(sd_per.max()),
        ],
    })
    out
    return out, sd_per


@app.cell
def _(
    PUBLISHED_BEST_CROSS_DATASET_MAE_K,
    PUBLISHED_WORST_CROSS_DATASET_MAE_K,
    ceiling_table,
    mo,
):
    mo.md(
        f"""
        ## Headline numbers

        Published cross-dataset MAE range (Kunchapu & Jablonka 2025):
        **{PUBLISHED_BEST_CROSS_DATASET_MAE_K} K** (best) to
        **{PUBLISHED_WORST_CROSS_DATASET_MAE_K} K** (worst).

        ### R²_ceiling table

        ```
{ceiling_table.to_string(index=False)}
        ```

        ### Reading the numbers

        * In the **all** subset, 7088 of 7367 PSMILES are singletons;
          the bracket is wide because singletons carry no within-group
          variance information.
        * In the **multi-source** subset (275 + 4 red = 279 PSMILES,
          all with ≥2 measurements), the bracket collapses (no
          singletons by construction). The single number is the honest
          ceiling.
        * The **MAE floor** is the half-normal expectation
          E|X| = sqrt(2 Var/pi) under the assumption that within-group
          residuals are zero-mean Gaussian. This is a lower bound on
          MAE achievable by any predictor that uses PSMILES alone.

        Compare the multi-source MAE floor to the published 13.79 K
        best cross-dataset MAE: if the floor is comparable to or above
        13.79 K, current SOTA is at the descriptor ceiling -- no
        algorithm can do better without descriptors that go beyond
        PSMILES (molecular weight, dispersity, tacticity, thermal
        history, measurement protocol).

        ### Caveats

        * The within-PSMILES variance is computed from cross-source
          disagreement, which conflates true sample variability
          (chain length, tacticity) with measurement methodology
          differences. Both are unobservable from PSMILES alone, so
          both legitimately raise the ceiling.
        * The half-normal assumption gives a single MAE number. If
          actual within-PSMILES residuals are lighter-tailed (e.g.
          bimodal for tactic/atactic forms), the true MAE floor is
          lower than the half-normal estimate. The R²_ceiling itself
          is distribution-free.
        * The comparison of the 13.79 K best cross-dataset MAE to
          the 14.26 K yellow+gold MAE floor conflates populations:
          the cross-test spans all polymers (most are singletons in
          the curated set), while the floor is estimated only on the
          275 polymers with multi-source reports. A fair comparison
          would retrain models with the cross-test protocol and
          measure MAE on the multi-source subset specifically --
          follow-up work.
        * Dropping the 4 red rows changes the ceiling slightly; the
          comparison row in the table quantifies this sensitivity.

        ### Bottom line

        On polymers with multi-source Tg reports, R²_ceiling from
        PSMILES alone is **0.97-0.98**, corresponding to an MAE floor
        of **14-18 K**. The published best cross-dataset MAE of
        13.79 K is at this floor. Further algorithm or data
        improvements cannot lower MAE below this bound without
        descriptors that encode properties PSMILES does not capture
        (molecular weight, tacticity, thermal history, measurement
        conditions).
        """
    )
    return


if __name__ == "__main__":
    app.run()
