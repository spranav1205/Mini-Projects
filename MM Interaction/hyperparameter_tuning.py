import os
import logging
import itertools
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from utils import (
    graph_from_molecule, PairedData, add_loops_to_data,
    GCNRegressor, GATRegressor, TransformerRegressor,
    evaluate, train_one_epoch, DEFAULT_NODE_DIM
)

log_file = "training.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),  # log to file
        logging.StreamHandler()                   # log to console too
    ]
)

logger = logging.getLogger("main")


# -------------------------------
# Data loading
# -------------------------------
def load_mol_from_cif_or_sdf(base_path, cutoff=2.5, max_atoms=200):
    cif = base_path + ".cif"
    sdf = base_path + ".sdf"
    mol = None
    print(f"Loading molecule from {base_path}")
    if os.path.exists(cif):
        from pymatgen.core import Structure
        structure = Structure.from_file(cif)
        m = Chem.RWMol()
        for site in structure:
            atom = Chem.Atom(site.specie.symbol)
            m.AddAtom(atom)
        if m.GetNumAtoms() > max_atoms:
            while m.GetNumAtoms() > max_atoms:
                m.RemoveAtom(m.GetNumAtoms()-1)
        conf = Chem.Conformer(m.GetNumAtoms())
        for idx, site in enumerate(structure[:m.GetNumAtoms()]):
            x, y, z = site.coords
            conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
        m.AddConformer(conf)
        mol = m

        print(f"Added atoms from {cif}:", mol.GetNumAtoms())

    elif os.path.exists(sdf):
        suppl = Chem.SDMolSupplier(sdf, removeHs=False)
        candidates = [m for m in suppl if m is not None]
        mol = candidates[0] if candidates else None
    return mol

def build_pairs_from_csv(csv_path, cif_dir, sdf_dir):
    df = pd.read_csv(csv_path)
    materials, drugs = [], []
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # --- Materials ---
    for i, cif_base in enumerate(df["material"]):
        mol = load_mol_from_cif_or_sdf(os.path.join(cif_dir, cif_base))
        if mol is None:
            logger.warning(f"Material {cif_base} could not be loaded")
            materials.append(None)
            continue
        g = graph_from_molecule(mol)
        g = add_loops_to_data(g)
        materials.append(g)
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1} / {len(df)} materials")

    # --- Drugs ---
    for i, sdf_base in enumerate(df["drug"]):
        mol = load_mol_from_cif_or_sdf(os.path.join(sdf_dir, sdf_base))
        if mol is None:
            logger.warning(f"Drug {sdf_base} could not be loaded")
            drugs.append(None)
            continue
        g = graph_from_molecule(mol)
        g = add_loops_to_data(g)
        drugs.append(g)
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1} / {len(df)} drugs")

    # --- Pair dataset ---
    dataset = []
    for i, row in df.iterrows():
        try:
            d1 = materials[i]
            d2 = drugs[i]
            if d1 is None or d2 is None:
                continue
            y = float(row["y"])
            weight = float(row["weight"]) if "weight" in df.columns else 1.0
            dataset.append(PairedData(d1, d2, y, weight))
        except Exception as e:
            logger.warning(f"Skipping row {i}: {e}")

    logger.info(f"Built dataset with {len(dataset)} pairs")
    return dataset

# -------------------------------
# Grid Search with K-Fold CV
# -------------------------------
def run_grid_search(dataset, device, epochs=100, k=5, save_path="grid_results_progress.csv"):
    results = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Slimmed grid (removed one non-essential param per model)
    model_grids = {
        "GCN": {
            "hidden_dim": [32, 64],
            "gnn_out_dim": [64, 96],
            "lr": [5e-3, 1e-3, 5e-4],  
        },
        # "GAT": {
        #     "gnn_out_dim": [64, 96],
        #     "heads": [2, 4],
        #     "lr": [1e-3, 7e-4],   
        # },
        "Transformer": {
            "d_model": [64, 128],
            "nhead": [4, 8],
            "num_layers": [2, 3],
            "lr": [3e-3, 1e-3 ,7e-4],   
        },
    }

    for model_name, param_grid in model_grids.items():
        keys, values = zip(*param_grid.items())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)

                if model_name == "GCN":
                    model = GCNRegressor(DEFAULT_NODE_DIM, **{k: params[k] for k in ["hidden_dim","gnn_out_dim"]})
                elif model_name == "GAT":
                    model = GATRegressor(DEFAULT_NODE_DIM, **{k: params[k] for k in ["gnn_out_dim","heads"]})
                else:
                    if params["d_model"] % params["nhead"] != 0:
                        continue
                    model = TransformerRegressor(
                        node_dim=DEFAULT_NODE_DIM,
                        d_model=params["d_model"],
                        nhead=params["nhead"],
                        num_layers=params["num_layers"],
                    )

                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)

                for ep in range(epochs):
                    train_one_epoch(model, train_subset, optimizer, device)

                score = evaluate(model, val_subset, device)
                fold_scores.append(score)
                logger.info(f"{model_name} fold {fold} | params {params} | RMSE {score:.4f}")

            if fold_scores:
                avg_rmse = sum(fold_scores) / len(fold_scores)
                results.append({"Model": model_name, "Hyperparameters": params, "RMSE": avg_rmse})
                
                # Save progress frequently
                df = pd.DataFrame(results)
                df.to_csv(save_path, index=False)
                logger.info(f"Updated results saved to {save_path}")

    return pd.DataFrame(results)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="trial_data.csv")
    parser.add_argument("--cif_dir", type=str, default="CIF_files")
    parser.add_argument("--sdf_dir", type=str, default="SDF_files")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--out", type=str, default="grid_rmse_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Building dataset...")
    dataset = build_pairs_from_csv(args.csv, args.cif_dir, args.sdf_dir)

    logger.info("Running grid search...")
    results_df = run_grid_search(dataset, device, epochs=args.epochs, k=args.k)

    print("\n=== Final Table: model | hyperparameters | RMSE (across folds) ===")
    print(results_df)
    results_df.to_csv(args.out, index=False)
    logger.info(f"Saved results to {args.out}")
