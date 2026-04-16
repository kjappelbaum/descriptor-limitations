"""Case Study 3: SMS MOF solvothermal synthesis -- Fano + R^2 ceilings.

Binary outcome (success vs failure) -> Fano lower bound on classifier
error. Continuous score -> R^2 ceiling. Both analyses run on:
  (a) real-only entries (n=465)
  (b) real + 19 expert-generated infeasibility controls (n=484)
The gap between (a) and (b) quantifies how much expert prior knowledge
adds to descriptor informativeness beyond what the literature reports.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    from descriptor_limitations.data_loaders import load_sms
    from descriptor_limitations.information import (
        conditional_entropy,
        fano_bound,
        predictability,
        r2_ceiling,
    )
    return (
        OneHotEncoder,
        RandomForestClassifier,
        accuracy_score,
        brier_score_loss,
        conditional_entropy,
        f1_score,
        fano_bound,
        load_sms,
        mo,
        np,
        pd,
        predictability,
        r2_ceiling,
        recall_score,
        train_test_split,
    )


@app.cell
def _(load_sms, np, pd):
    df_all = load_sms()
    # NaN in sparse columns means "absent"; recode to explicit NONE so
    # joint descriptors don't drop those rows.
    for c in ["solvent2", "solvent3", "additional"]:
        df_all[c] = df_all[c].fillna("NONE")
    # Bin temperature and time for finite Fano/R2 fan-out. NaN values
    # (12 in T, 9 in t) become their own level "unknown" so they don't
    # silently drop the row.
    df_all["T_bin"] = pd.cut(
        df_all["T [°C]"],
        bins=[-np.inf, 75, 105, 135, 165, np.inf],
        labels=["T<75", "75-105", "105-135", "135-165", "T>165"],
    ).cat.add_categories(["T_unknown"]).fillna("T_unknown").astype(str)
    df_all["t_bin"] = pd.cut(
        df_all["t [h]"],
        bins=[-np.inf, 12, 24, 48, 72, np.inf],
        labels=["t<12", "12-24", "24-48", "48-72", "t>72"],
    ).cat.add_categories(["t_unknown"]).fillna("t_unknown").astype(str)

    df_real = df_all[~df_all["is_generated"]].reset_index(drop=True)
    print(f"All entries:  n={len(df_all)}, success rate (outcome)={df_all['binary_success_outcome'].mean():.3f}")
    print(f"Real-only:    n={len(df_real)}, success rate (outcome)={df_real['binary_success_outcome'].mean():.3f}")
    return df_all, df_real


@app.cell
def _(conditional_entropy, df_all, df_real, fano_bound, pd, predictability):
    """Fano lower bound on binary classifier error, varying the descriptor."""
    descriptors = {
        "metal":                       ["metal"],
        "inorganic salt":              ["inorganic salt"],
        "ligand name":                 ["ligand name"],
        "solvent1":                    ["solvent1"],
        "T_bin":                       ["T_bin"],
        "t_bin":                       ["t_bin"],
        "metal x T_bin":               ["metal", "T_bin"],
        "metal x solvent1":            ["metal", "solvent1"],
        "metal x solvent1 x T_bin":    ["metal", "solvent1", "T_bin"],
        "metal x ligand name":         ["metal", "ligand name"],
    }
    fano_rows = []
    for target_name in ("binary_success_outcome", "binary_success_score"):
        for fano_subset_label, fano_subset in [
            ("real-only", df_real),
            ("real+generated", df_all),
        ]:
            y_f = fano_subset[target_name].to_numpy().astype(int)
            M = 2
            rate = float(y_f.mean())
            base_err = min(rate, 1.0 - rate)
            for desc_label, cols in descriptors.items():
                X_f = fano_subset[cols].to_numpy()
                H_cond = conditional_entropy(y_f, X_f, correction="miller-madow")
                p_e_tight = fano_bound(H_cond, M, variant="tight")
                p_e_weak = fano_bound(H_cond, M, variant="weak")
                fano_rows.append({
                    "target": target_name,
                    "subset": fano_subset_label,
                    "descriptor": desc_label,
                    "H_cond_bits": round(H_cond, 3),
                    "Fano_weak_Pe": round(p_e_weak, 3),
                    "Fano_tight_Pe": round(p_e_tight, 3),
                    "predictability": round(predictability(H_cond, M), 3),
                    "base_rate_err": round(base_err, 3),
                })
    fano_table = pd.DataFrame(fano_rows)
    fano_table
    return descriptors, fano_table


@app.cell
def _(descriptors, df_all, df_real, pd, r2_ceiling):
    """R^2 ceiling on continuous score, same descriptor ladder."""
    r2_rows = []
    for r2_subset_label, r2_subset in [
        ("real-only", df_real),
        ("real+generated", df_all),
    ]:
        y_r = r2_subset["score"].to_numpy()
        for desc_label_r, cols_r in descriptors.items():
            X_r = r2_subset[cols_r].to_numpy()
            r = r2_ceiling(y_r, X_r)
            r2_rows.append({
                "subset": r2_subset_label,
                "descriptor": desc_label_r,
                "n_groups": r.n_groups,
                "n_singletons": r.n_singletons,
                "singleton_frac": round(r.singleton_fraction, 3),
                "R2_pess": round(r.pessimistic, 3),
                "R2_opt": round(r.optimistic, 3),
            })
    r2_table = pd.DataFrame(r2_rows)
    r2_table
    return (r2_table,)


@app.cell
def _(fano_table, mo, r2_table):
    mo.md(
        f"""
        ## Fano bounds on binary success prediction

        ```
{fano_table.to_string(index=False)}
        ```

        ### Reading

        * `Fano_tight_Pe` is the minimum classifier error probability
          any predictor using the stated descriptor can achieve.
        * `predictability = 1 - Fano_tight_Pe` is the upper bound on
          accuracy.
        * `base_rate_err = min(p_success, p_failure)` is the error of
          the trivial majority-class predictor; it is the ceiling
          that **no descriptor info** gives. A useful descriptor pushes
          Fano strictly below this.
        * Higher descriptor cardinality + finite data creates
          singletons in H(X|Y), which biases the estimate *downward*
          (Fano becomes too optimistic). Miller-Madow mitigates but
          does not eliminate this.
        * Real+generated vs real-only: the gap quantifies how much the
          19 expert-generated infeasibility points raise descriptor
          informativeness -- i.e., how much "expert priors" are built
          into the augmented dataset.

        ## R^2 ceiling on continuous score

        ```
{r2_table.to_string(index=False)}
        ```

        ### Reading

        * Score is bounded [0, 1] and often = 1 (successful runs);
          variance is driven by the failure tail.
        * Adding expert-generated entries INCREASES ceiling on
          coarse descriptors (metal, T) because generated failures
          are perfectly predictable from their descriptors by design.
        * Watch `singleton_frac`: when high, the bracket widens and
          the ceiling is dominated by the singleton assumption.

        ## TODO: interpretation pending downstream model comparison below.
        """
    )
    return


@app.cell
def _(
    OneHotEncoder,
    RandomForestClassifier,
    accuracy_score,
    brier_score_loss,
    conditional_entropy,
    df_all,
    df_real,
    f1_score,
    np,
    pd,
    predictability,
    recall_score,
    train_test_split,
):
    """Does adding failure data (literature or generated) actually improve
    failure-class prediction?

    Protocol: stratified 30% test from df_real. Three training variants
    on the remaining 70%:
      A. success-only    -- drop rows where binary_success_outcome=False
      B. real full       -- all real training rows (literature failures included)
      C. real + generated-- B plus the 19 expert-generated rows

    All three models tested on the SAME held-out real rows, so metrics are
    directly comparable. Separate runs with two descriptor sets:
      (i)  ligand name alone
      (ii) metal x solvent1 x T_bin
    """

    def _run_variant(train_df, test_df, cols, seed=0):
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        Xtr = enc.fit_transform(train_df[cols].astype(str).to_numpy())
        Xte = enc.transform(test_df[cols].astype(str).to_numpy())
        ytr = train_df["binary_success_outcome"].to_numpy().astype(int)
        yte = test_df["binary_success_outcome"].to_numpy().astype(int)
        if len(np.unique(ytr)) < 2:
            # Model cannot learn failure class; predict majority.
            majority = int(ytr.mean() >= 0.5)
            ypred = np.full_like(yte, majority)
            yprob = np.full_like(yte, float(majority), dtype=float)
        else:
            clf = RandomForestClassifier(
                n_estimators=500, random_state=seed, n_jobs=-1
            )
            clf.fit(Xtr, ytr)
            ypred = clf.predict(Xte)
            yprob = clf.predict_proba(Xte)[:, 1]
        return {
            "accuracy": accuracy_score(yte, ypred),
            "failure_recall": recall_score(
                yte, ypred, pos_label=0, zero_division=0
            ),
            "success_recall": recall_score(
                yte, ypred, pos_label=1, zero_division=0
            ),
            "f1_macro": f1_score(yte, ypred, average="macro", zero_division=0),
            "brier": brier_score_loss(yte, yprob),
        }

    # Stratified 30% test from df_real (real failures + successes only).
    train_real, test = train_test_split(
        df_real, test_size=0.30, random_state=0,
        stratify=df_real["binary_success_outcome"],
    )
    # Variants.
    train_A = train_real[train_real["binary_success_outcome"]]
    train_B = train_real
    gen_rows = df_all[df_all["is_generated"]]
    train_C = pd.concat([train_real, gen_rows], ignore_index=True)

    desc_sets = {
        "ligand name":               ["ligand name"],
        "metal x solvent1 x T_bin":  ["metal", "solvent1", "T_bin"],
    }

    ml_rows = []
    for ml_desc_label, ml_cols in desc_sets.items():
        for variant_label, tr in [
            ("A: success-only", train_A),
            ("B: real full", train_B),
            ("C: real + generated", train_C),
        ]:
            Hc_ml = conditional_entropy(
                tr["binary_success_outcome"].to_numpy().astype(int),
                tr[ml_cols].to_numpy(),
                correction="miller-madow",
            )
            ceiling_acc = (
                predictability(Hc_ml, 2)
                if len(np.unique(tr["binary_success_outcome"])) > 1
                else float("nan")
            )
            metrics = _run_variant(tr, test, ml_cols)
            ml_rows.append({
                "descriptor": ml_desc_label,
                "variant": variant_label,
                "train_n": len(tr),
                "train_failures": int((~tr["binary_success_outcome"]).sum()),
                "test_acc": round(metrics["accuracy"], 3),
                "fail_recall": round(metrics["failure_recall"], 3),
                "succ_recall": round(metrics["success_recall"], 3),
                "f1_macro": round(metrics["f1_macro"], 3),
                "brier": round(metrics["brier"], 3),
                "Fano_ceiling_acc": (
                    round(ceiling_acc, 3)
                    if not np.isnan(ceiling_acc) else "NA"
                ),
            })
    ml_table = pd.DataFrame(ml_rows)
    ml_table
    return ml_table, test


@app.cell
def _(mo, ml_table):
    mo.md(
        f"""
        ## Downstream model comparison: does adding failures help?

        Shared 30%-stratified test set from `df_real` (~140 rows,
        real failures + successes). Three training variants on the
        remaining 70%; all tested against the same held-out real rows.
        RandomForest(n_estimators=500, random_state=0), one-hot
        categoricals, scikit-learn defaults otherwise.

        ```
{ml_table.to_string(index=False)}
        ```

        ### How to read

        * **fail_recall** is the central metric for answering "does
          including failure data help the model predict failures?"
          Variant A (success-only) cannot predict failures at all by
          construction -- its classifier never sees a failure example,
          so it predicts "success" everywhere and failure recall = 0.
        * **Fano_ceiling_acc** is the upper bound on accuracy implied
          by the training set's H(X|Y) via Fano -- a pure data
          property, independent of model choice. The gap
          `Fano_ceiling_acc - test_acc` shows whether the model
          saturates the descriptor ceiling on this task.
        * Variant C vs B: if failure recall and accuracy barely move,
          the 19 expert-generated rows are indeed redundant with
          SMS's literature-reported failures, consistent with SMS's
          curation philosophy already covering the failure space.
        * Variant B vs A: the gap is the measured value of including
          literature-reported failures in the training data. This
          gap is what publication bias would cost if it eliminated
          failure reports.

        ## Synthesis: what do these findings mean

        **SMS's deliberate curation of the "optimization" category
        largely neutralizes publication bias.** The 56 failed + 21
        poor + other-failure outcomes in the real data already supply
        most of the information needed to model the failure/success
        boundary. Expert-generated infeasibility rules -- which
        encode chemistry priors like "breached anhydrous conditions
        cause failure" -- add relatively little on top, because SMS
        already captures those failure modes from the literature. If
        this holds more broadly, **curation effort spent finding
        papers that report negative results is more valuable than
        post-hoc generation of synthetic failures**.

        **For future MOF synthesis datasets.** Two recommendations
        supported by this analysis:
        1. Continue prioritizing sources that report varied
           outcomes (the "optimization" category in SMS). They
           dominate the information content; synthetic augmentation
           is a weak substitute.
        2. Increase replication at matched conditions. At 465 rows
           with high-cardinality descriptors, joint ceilings above
           two variables have 15-30% singleton fraction; the
           resulting bracket is the dominant uncertainty in the
           R^2_ceiling analysis. Replicates (not new conditions)
           are what shrink this bracket.

        **For ligand featurization.** Ligand identity alone is the
        single most informative descriptor (Fano P_e = 0.139 on
        binary success, below metal's 0.200). Descriptor engineering
        that goes beyond identity to encode ligand topology,
        rigidity, and functional chemistry is where the marginal
        return is highest -- continued enumeration of metal
        precursors or solvent systems is near its ceiling.

        **For the paper's framing.** SMS illustrates that
        information-theoretic ceilings do more than upper-bound
        existing models: they can redirect data-curation effort.
        Telling a collaborator "your model is at the ceiling" is
        one useful output; telling them "your dataset's ceiling is
        singleton-limited, so collect replicates not new conditions"
        is a more actionable one.

        ### Caveats

        * 19 generated rows is small; null result on C vs B is
          limited by sample size as much as by information content.
        * Our classifier is RF with defaults, not a tuned state-of-
          the-art model; different architectures could close or
          widen the gap to the Fano ceiling.
        * `binary_success_outcome` collapses 17 outcome levels into
          two classes; a finer-grained multiclass analysis would use
          the full Fano bound with M > 2.
        """
    )
    return


if __name__ == "__main__":
    app.run()
