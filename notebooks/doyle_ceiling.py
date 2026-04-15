"""Case Study 1: Doyle Buchwald-Hartwig HTE -- R^2 ceiling analysis.

Computes R^2_ceiling at every descriptor resolution (singleton-bracketed)
and reproduces the published random forest R^2 (0.92 random split,
0.83 leave-one-additive-out) with one-hot encoded categorical features.

Run interactively:
    uv run marimo edit notebooks/doyle_ceiling.py
Run headless (smoke test):
    uv run marimo run notebooks/doyle_ceiling.py
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import itertools

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    from descriptor_limitations.data_loaders import load_doyle
    from descriptor_limitations.information import r2_ceiling

    PUBLISHED_R2_RANDOM_SPLIT = 0.92  # Ahneman 2018, RF + DFT, 70/30 split
    PUBLISHED_R2_LOO_ADDITIVE = 0.83  # Ahneman 2018, leave-one-additive-out
    return (
        OneHotEncoder,
        PUBLISHED_R2_LOO_ADDITIVE,
        PUBLISHED_R2_RANDOM_SPLIT,
        RandomForestRegressor,
        itertools,
        load_doyle,
        np,
        pd,
        plt,
        r2_ceiling,
        r2_score,
        train_test_split,
    )


@app.cell
def _(load_doyle):
    df = load_doyle()
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(
        f"Factor cardinalities: base={df['base'].nunique()} "
        f"ligand={df['ligand'].nunique()} "
        f"halide={df['halide'].nunique()} "
        f"additive={df['additive'].nunique()}"
    )
    print(f"Yield: min={df['yield'].min():.2f} mean={df['yield'].mean():.2f} "
          f"max={df['yield'].max():.2f}")
    df.head()
    return (df,)


@app.cell
def _(df, itertools, pd, r2_ceiling):
    """R^2_ceiling at every subset of {base, ligand, halide, additive},
    plus plate alone (batch-effect diagnostic)."""
    factors = ["base", "ligand", "halide", "additive"]
    rows = []
    # All non-empty subsets, finest (4-way) to coarsest (1-way).
    for k in range(len(factors), 0, -1):
        for subset in itertools.combinations(factors, k):
            g = df[list(subset)].to_numpy()
            r = r2_ceiling(df["yield"].to_numpy(), g)
            rows.append({
                "descriptor": " x ".join(subset),
                "n_vars": len(subset),
                "n_groups": r.n_groups,
                "n_singletons": r.n_singletons,
                "singleton_frac": r.singleton_fraction,
                "R2_ceiling_optimistic": r.optimistic,
                "R2_ceiling_pessimistic": r.pessimistic,
                "bracket_width": r.optimistic - r.pessimistic,
            })

    # Plate alone (batch effect check).
    r_plate = r2_ceiling(df["yield"].to_numpy(), df["plate"].to_numpy())
    rows.append({
        "descriptor": "plate",
        "n_vars": 1,
        "n_groups": r_plate.n_groups,
        "n_singletons": r_plate.n_singletons,
        "singleton_frac": r_plate.singleton_fraction,
        "R2_ceiling_optimistic": r_plate.optimistic,
        "R2_ceiling_pessimistic": r_plate.pessimistic,
        "bracket_width": r_plate.optimistic - r_plate.pessimistic,
    })

    ceiling_table = pd.DataFrame(rows)
    ceiling_table = ceiling_table.sort_values(
        ["n_vars", "R2_ceiling_pessimistic"], ascending=[False, False]
    ).reset_index(drop=True)
    ceiling_table
    return ceiling_table, factors


@app.cell
def _(PUBLISHED_R2_RANDOM_SPLIT, ceiling_table, plt):
    """Bracket plot: R^2_ceiling at each descriptor resolution."""
    tbl = ceiling_table.iloc[::-1]  # plot coarsest at bottom
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(tbl) + 1.5))
    y_pos = range(len(tbl))
    for i, (_, row) in enumerate(tbl.iterrows()):
        ax.plot(
            [row["R2_ceiling_pessimistic"], row["R2_ceiling_optimistic"]],
            [i, i],
            "-", color="C0", lw=2,
        )
        ax.plot(row["R2_ceiling_pessimistic"], i, "|", color="C0", ms=10)
        ax.plot(row["R2_ceiling_optimistic"], i, "|", color="C0", ms=10)
    ax.axvline(
        PUBLISHED_R2_RANDOM_SPLIT, color="C3", ls="--",
        label=f"Published R^2 = {PUBLISHED_R2_RANDOM_SPLIT} (random split)",
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tbl["descriptor"], fontsize=8)
    ax.set_xlabel("R^2_ceiling (bracket: pessimistic to optimistic)")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Doyle HTE: R^2_ceiling vs descriptor resolution")
    fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _(OneHotEncoder, RandomForestRegressor, df, r2_score, train_test_split):
    """Reproduce published R^2 = 0.92: one-hot RF, 70/30 random split."""
    X_cat = df[["base", "ligand", "halide", "additive"]].astype(str).to_numpy()
    y = df["yield"].to_numpy()

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(X_cat)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    r2_random_split = r2_score(y_te, rf.predict(X_te))
    print(f"One-hot RF, 70/30 random split: R^2 = {r2_random_split:.3f}")
    print(f"Published (Ahneman 2018, DFT RF): R^2 = 0.92")
    return X, X_cat, enc, r2_random_split, rf, y


@app.cell
def _(
    OneHotEncoder,
    RandomForestRegressor,
    df,
    np,
    r2_score,
):
    """Reproduce leave-one-additive-out R^2 = 0.83."""
    additives_loo = df["additive"].unique()
    fold_r2s = []
    y_loo = df["yield"].to_numpy()
    for held_out in additives_loo:
        tr_mask = df["additive"].to_numpy() != held_out
        te_mask = ~tr_mask
        if te_mask.sum() < 5:
            continue
        enc_loo = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        Xtr_loo = enc_loo.fit_transform(
            df.loc[tr_mask, ["base", "ligand", "halide", "additive"]]
            .astype(str).to_numpy()
        )
        Xte_loo = enc_loo.transform(
            df.loc[te_mask, ["base", "ligand", "halide", "additive"]]
            .astype(str).to_numpy()
        )
        rf_loo = RandomForestRegressor(
            n_estimators=500, random_state=0, n_jobs=-1
        )
        rf_loo.fit(Xtr_loo, y_loo[tr_mask])
        fold_r2s.append(r2_score(y_loo[te_mask], rf_loo.predict(Xte_loo)))
    r2_loo_additive_mean = float(np.mean(fold_r2s))
    print(
        f"One-hot RF, LOO additive: mean R^2 = {r2_loo_additive_mean:.3f} "
        f"(over {len(fold_r2s)} folds)"
    )
    print(f"Published (Ahneman 2018, DFT RF, LOO additive): R^2 = 0.83")
    return fold_r2s, r2_loo_additive_mean


@app.cell
def _(
    PUBLISHED_R2_LOO_ADDITIVE,
    PUBLISHED_R2_RANDOM_SPLIT,
    ceiling_table,
    pd,
    r2_loo_additive_mean,
    r2_random_split,
):
    """Comparison: published / reproduced R^2 vs ceilings at relevant resolutions."""
    full_joint = ceiling_table[ceiling_table["descriptor"] == "base x ligand x halide x additive"].iloc[0]
    no_additive = ceiling_table[ceiling_table["descriptor"] == "base x ligand x halide"].iloc[0]
    summary = pd.DataFrame([
        {
            "comparison": "Random split (RF on full descriptor)",
            "published_R2": PUBLISHED_R2_RANDOM_SPLIT,
            "reproduced_R2_one_hot": r2_random_split,
            "ceiling_pessimistic": full_joint["R2_ceiling_pessimistic"],
            "ceiling_optimistic": full_joint["R2_ceiling_optimistic"],
        },
        {
            "comparison": "LOO-additive (model never sees held-out additive)",
            "published_R2": PUBLISHED_R2_LOO_ADDITIVE,
            "reproduced_R2_one_hot": r2_loo_additive_mean,
            "ceiling_pessimistic": no_additive["R2_ceiling_pessimistic"],
            "ceiling_optimistic": no_additive["R2_ceiling_optimistic"],
        },
    ])
    summary
    return (summary,)


@app.cell
def _():
    # TODO: interpret after running. Placeholder cell.
    # Discuss with collaborator before writing conclusions.
    """## TODO: interpret after seeing numbers."""
    return


if __name__ == "__main__":
    app.run()
