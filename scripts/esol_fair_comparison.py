"""Fair same-population ESOL model comparison.

Trains GBR on AqSolDB median logS (excluding source G = ESOL's own data)
for the 962 ESOL compounds with ≥2 remaining non-G measurements.
Morgan fingerprints (r=2, 2048 bits) as features. 5-fold CV.
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from descriptor_limitations.data_loaders import load_aqsoldb_sources

RDLogger.DisableLog("rdApp.*")


def morgan_fp(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def main():
    aq = load_aqsoldb_sources()
    esol = pd.read_csv("/tmp/esol.csv")
    esol["ik"] = esol["smiles"].apply(
        lambda s: (m := Chem.MolFromSmiles(s)) and Chem.MolToInchiKey(m)
    )

    # Exclude source G (= ESOL)
    aq_no_g = aq[aq["source"] != "G"]
    esol_iks = set(esol["ik"].dropna())
    aq_esol = aq_no_g[aq_no_g["InChIKey"].isin(esol_iks)]

    # Keep only ESOL compounds with ≥2 non-G measurements
    per = aq_esol.groupby("InChIKey").size()
    multi_iks = per[per >= 2].index
    aq_multi = aq_esol[aq_esol["InChIKey"].isin(multi_iks)]

    # Median logS per compound (training label)
    medians = aq_multi.groupby("InChIKey").agg(
        logS_median=("Solubility", "median"),
        n_sources=("Solubility", "size"),
    ).reset_index()

    # Get a canonical SMILES for each InChIKey (from ESOL)
    ik_to_smi = esol.set_index("ik")["smiles"].to_dict()
    medians["smiles"] = medians["InChIKey"].map(ik_to_smi)
    medians = medians.dropna(subset=["smiles"])

    # Compute fingerprints
    fps = []
    valid_idx = []
    for i, smi in enumerate(medians["smiles"]):
        fp = morgan_fp(smi)
        if fp is not None:
            fps.append(np.array(fp))
            valid_idx.append(i)
    X = np.array(fps)
    medians = medians.iloc[valid_idx].reset_index(drop=True)
    y = medians["logS_median"].to_numpy()

    print(f"Compounds for fair comparison: {len(medians)}")
    print(f"Feature matrix: {X.shape}")
    print()

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmse, fold_mae = [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        model = GradientBoostingRegressor(
            n_estimators=500, max_depth=5, random_state=0
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        rmse = np.sqrt(mean_squared_error(y[te], pred))
        mae = mean_absolute_error(y[te], pred)
        fold_rmse.append(rmse)
        fold_mae.append(mae)
        print(f"  Fold {fold}: RMSE={rmse:.3f} MAE={mae:.3f}")

    print()
    print(f"5-fold CV:  RMSE = {np.mean(fold_rmse):.3f} ± {np.std(fold_rmse):.3f}")
    print(f"            MAE  = {np.mean(fold_mae):.3f} ± {np.std(fold_mae):.3f}")
    print()
    print("=== Comparison ===")
    print(f"Descriptor ceiling RMSE floor:  0.255 LogS")
    print(f"GBR (this script, 5-fold CV):   {np.mean(fold_rmse):.3f} LogS")
    print(f"D-MPNN (published, ESOL split): 0.555 LogS")
    print(f"Headroom (GBR → floor):         {np.mean(fold_rmse) - 0.255:.3f} LogS")


if __name__ == "__main__":
    main()
