"""Case Study 4: MPEA yield strength -- family-stratified ceilings.

Composition-only ceilings on the MPEA dataset (Borg et al. 2020) are
very different for Cantor-family FCC alloys vs refractory BCC alloys,
and the two sub-populations are differentially represented in the
grain-size-reported subset. Pooling conflates the physics regimes.

This notebook runs the ceiling ladder (composition -> +T_bin ->
+grain_bin) stratified by family. Hall-Petch is resolvable only on
the Cantor subset where enough rows report grain size.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from descriptor_limitations.data_loaders import load_mpea
    from descriptor_limitations.information import r2_ceiling
    return load_mpea, mo, np, pd, r2_ceiling


@app.cell
def _(load_mpea, np, pd):
    raw = load_mpea()
    raw["T_bin"] = pd.cut(
        raw["test_T_C"],
        bins=[-np.inf, 50, 500, 1000, np.inf],
        labels=["RT", "warm", "hot", "very_hot"],
    ).cat.add_categories(["T_unknown"]).fillna("T_unknown").astype(str)
    raw["grain_bin"] = pd.cut(
        np.log10(raw["grain_size_um"].where(raw["grain_size_um"] > 0)),
        bins=[-np.inf, 0, 1, 2, np.inf],
        labels=["<1um", "1-10um", "10-100um", ">100um"],
    ).astype(str)

    # Alloy family classification.
    # Cantor family: 3+ of {Co, Cr, Fe, Mn, Ni}.
    # Refractory:    contains at least one of {W, Mo, Nb, Ta, Hf, Re}
    #                AND is not Cantor. Disjoint by construction.
    CANTOR_ELEMS = ["Co", "Cr", "Fe", "Mn", "Ni"]
    REFRAC_ELEMS = ["W", "Mo", "Nb", "Ta", "Hf", "Re"]

    def is_cantor(formula: str) -> bool:
        return sum(e in formula for e in CANTOR_ELEMS) >= 3

    def is_refractory(formula: str) -> bool:
        return any(e in formula for e in REFRAC_ELEMS)

    raw["family"] = "other"
    raw.loc[raw["formula"].apply(is_cantor), "family"] = "cantor"
    raw.loc[
        raw["formula"].apply(lambda f: is_refractory(f) and not is_cantor(f)),
        "family",
    ] = "refractory"

    ys_subset = raw[raw["YS_MPa"].notna()].reset_index(drop=True)
    hp_subset = raw[
        raw["YS_MPa"].notna() & raw["grain_size_um"].notna()
    ].reset_index(drop=True)

    print(f"Full YS subset:         n={len(ys_subset)}")
    print(f"Hall-Petch subset:      n={len(hp_subset)}")
    print(f"YS (MPa): min={ys_subset['YS_MPa'].min():.0f} "
          f"median={ys_subset['YS_MPa'].median():.0f} "
          f"max={ys_subset['YS_MPa'].max():.0f}")
    print()
    print("Family distribution in YS subset:")
    print(ys_subset["family"].value_counts().to_string())
    print()
    print("Family distribution in Hall-Petch subset:")
    print(hp_subset["family"].value_counts().to_string())
    return hp_subset, raw, ys_subset


@app.cell
def _(hp_subset, pd, r2_ceiling, ys_subset):
    """Stratified ceiling ladder.

    For each family, compute:
      (i)  composition alone (ys_subset)
      (ii) composition + T_bin (ys_subset)
      (iii) composition + T_bin (grain subset only)
      (iv) composition + T_bin + grain_bin (grain subset only)
    Rows (iii) and (iv) are computed on the SAME subset so the jump
    (iv)-(iii) is the apples-to-apples Hall-Petch contribution.
    """
    def _row(df_src, cols, desc, family_label, subset_label):
        r = r2_ceiling(df_src["YS_MPa"].to_numpy(), df_src[cols].to_numpy())
        return {
            "family": family_label,
            "subset": subset_label,
            "n": len(df_src),
            "descriptor": desc,
            "n_groups": r.n_groups,
            "n_singletons": r.n_singletons,
            "singleton_frac": round(r.singleton_fraction, 3),
            "R2_pess": round(r.pessimistic, 3),
            "R2_opt": round(r.optimistic, 3),
        }

    rows = []
    for fam in ["pooled", "cantor", "refractory", "other"]:
        ys_fam = ys_subset if fam == "pooled" else ys_subset[ys_subset["family"] == fam]
        hp_fam = hp_subset if fam == "pooled" else hp_subset[hp_subset["family"] == fam]
        if len(ys_fam) == 0:
            continue
        rows.append(_row(ys_fam, ["formula"], "formula", fam, "full YS"))
        rows.append(_row(ys_fam, ["formula", "T_bin"], "formula x T_bin", fam, "full YS"))
        if len(hp_fam) >= 20:
            rows.append(_row(hp_fam, ["formula", "T_bin"],
                             "formula x T_bin", fam, "grain-reported"))
            rows.append(_row(hp_fam, ["formula", "T_bin", "grain_bin"],
                             "formula x T_bin x grain_bin", fam, "grain-reported"))
    ladder_table = pd.DataFrame(rows)
    ladder_table
    return (ladder_table,)


@app.cell
def _(ladder_table, mo):
    mo.md(
        f"""
        ## Family-stratified ceiling ladder

        Rows marked "grain-reported" are computed on the matched grain-
        size-available subset of each family. The Hall-Petch jump is the
        change from "formula x T_bin" to "formula x T_bin x grain_bin"
        WITHIN a family's grain-reported subset (apples-to-apples).

        ```
{ladder_table.to_string(index=False)}
        ```

        ### Key findings

        1. **Pooled ceilings mask family-specific physics.** Pooled
           composition ceiling is ~[0.41, 0.64]; Cantor's is
           ~[0.49, 0.80]; refractory's is ~[0.33, 0.48]. Reporting a
           single "MPEA benchmark R^2" averages over two regimes with
           very different descriptor informativeness.

        2. **Temperature dominates refractory ceilings but is marginal
           for Cantor.** Adding T_bin to refractory composition: pess
           0.33 -> 0.61 (+0.28). To Cantor composition: 0.49 -> 0.51
           (+0.02). This is structural: Cantor alloys are
           predominantly tested at room temperature in this dataset;
           refractory alloys are studied across a wide T range for
           high-T applications.

        3. **Hall-Petch is real and quantifiable on Cantor
           (apples-to-apples).** On the Cantor grain-reported subset
           (n=160): composition+T gives [0.49, 0.71]; adding grain_bin
           gives [0.58, 0.91]. Jump: pess +0.09, opt +0.20. This is
           the "Hall-Petch in R^2 units" the framework was supposed to
           measure.

        4. **Refractory grain-reported subset is too small (n=42) to
           resolve Hall-Petch.** Singleton fraction 0.57 at
           composition+T; the bracket is dominated by singleton
           uncertainty, not by variance estimation. Need more grain-
           size-reporting for refractory alloys in future curations.

        5. **The grain-reported subset is not representative of the
           full dataset, in a way driven by sub-domain, not reporting
           rigor.** 77% of grain-reported rows are Cantor-family;
           65% of grain-absent rows are refractory. A check on
           source-article concentration (55 articles for grain-reported,
           154 for grain-absent) ruled out lab-concentration as the
           cause, and the physical explanation (Cantor alloys studied
           with standard metallography at RT; refractory studied at
           high T with microstructural evolution) makes the sub-domain
           account plausible.

        ## What this case study adds to the paper

        * A cross-domain (metallurgy) application of the framework.
        * A concrete case where **pooled benchmarks conflate physical
          regimes**, demonstrating that per-dataset ceiling reporting
          should be stratified when the data spans distinct physics.
        * A clean, apples-to-apples **Hall-Petch jump of pess +0.09 /
          opt +0.20 on Cantor alloys**, measured without any ML model.

        ## What this case study does NOT say

        * We do **not** claim "reporting grain size makes the dataset
          better." The grain-reported subset differs from the
          grain-absent subset because different alloy families have
          different reporting conventions, not because reporting
          grain size causes rigor. See Appendix analysis on
          source-article and alloy-family concentration.
        * We do **not** quantify Hall-Petch on refractory alloys; the
          dataset is too sparse on that axis for the current
          framework to resolve it.

        ### Caveats

        * Grain size is reported inconsistently across sources
          (method: EBSD, optical, linear intercept). Binning to
          decadal log ranges mitigates but does not eliminate this.
        * T_bin is coarse; finer bins would tighten the +T jump but
          raise singleton fraction.
        * Family classification is a heuristic (3+ of Cantor elements;
          at least one refractory and not Cantor). A small `other`
          bucket (29 rows) is not analyzed.
        """
    )
    return


if __name__ == "__main__":
    app.run()
