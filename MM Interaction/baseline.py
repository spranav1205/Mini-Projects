import os
import numpy as np
import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from rdkit import Chem

logger = logging.getLogger(__name__)

# Define atomic properties manually
ATOM_PROPS = {
    "H": {"en": 2.20, "mass": 1.008},
    "C": {"en": 2.55, "mass": 12.011},
    "N": {"en": 3.04, "mass": 14.007},
    "O": {"en": 3.44, "mass": 15.999},
    "F": {"en": 3.98, "mass": 18.998},
    "P": {"en": 2.19, "mass": 30.974},
    "S": {"en": 2.58, "mass": 32.06},
    "Cl": {"en": 3.16, "mass": 35.45},
    "Br": {"en": 2.96, "mass": 79.904},
    "I": {"en": 2.66, "mass": 126.90},
    # add more as needed
}

def load_mol_from_cif_or_sdf(base_path, cutoff=2.5, max_atoms=200):
    cif = base_path + ".cif"
    sdf = base_path + ".sdf"
    mol = None
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

def molecule_to_feature_vector(mol):
    """
    Generate a generic feature vector for a molecule
    mol: pymatgen/ASE/RDKit molecule object (from load_mol_from_cif_or_sdf)
    """
    if mol is None:
        return None

    atom_counts = {}
    en_values = []

    # Iterate over atoms
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        props = ATOM_PROPS.get(symbol, {"en": 0.0, "mass": 0.0})
        en_values.append(props["en"])
        atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

    # Build feature vector
    all_symbols = sorted(ATOM_PROPS.keys())  # consistent ordering
    counts_vec = [atom_counts.get(sym, 0) for sym in all_symbols]

    if en_values:
        max_en = max(en_values)
        min_en = min(en_values)
        mean_en = np.mean(en_values)
    else:
        max_en = min_en = mean_en = 0.0

    # Concatenate counts and electronegativity stats
    feature_vec = np.array(counts_vec + [max_en, min_en, mean_en], dtype=np.float32)
    return feature_vec

def build_paired_features(csv_path, cif_dir, sdf_dir):
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    paired_dataset = []

    for i, row in df.iterrows():
        # Load molecules
        mat_mol = load_mol_from_cif_or_sdf(os.path.join(cif_dir, row["material"]))
        drug_mol = load_mol_from_cif_or_sdf(os.path.join(sdf_dir, row["drug"]))

        if mat_mol is None or drug_mol is None:
            logger.warning(f"Skipping row {i}: molecule missing")
            continue

        # Generate feature vectors
        mat_vec = molecule_to_feature_vector(mat_mol)
        drug_vec = molecule_to_feature_vector(drug_mol)

        if mat_vec is None or drug_vec is None:
            continue

        # Concatenate feature vectors
        paired_vec = np.concatenate([mat_vec, drug_vec])

        # Target
        y = float(row["y"])
        weight = float(row["weight"]) if "weight" in df.columns else 1.0

        paired_dataset.append((paired_vec, y, weight))

    logger.info(f"Built paired dataset with {len(paired_dataset)} samples")
    return paired_dataset

paired_dataset = build_paired_features("trial_dataCopy.csv", "data/CIF_files/", "data/SDF_files/")

# Suppose paired_dataset is your output from build_paired_features
# Each element: (paired_vec, y, weight)
X = np.array([vec for vec, _, _ in paired_dataset], dtype=np.float32)
y = np.array([y_val for _, y_val, _ in paired_dataset], dtype=np.float32)
weights = np.array([w for _, _, w in paired_dataset], dtype=np.float32)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y, weights):
    mses = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train, w_test = weights[train_idx], weights[test_idx]

        model.fit(X_train, y_train, sample_weight=w_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, sample_weight=w_test)
        mses.append(mse)
    return np.mean(mses), np.std(mses)

# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_mean_mse, rf_std_mse = evaluate_model(rf_model, X, y, weights)
print(f"Random Forest MSE: {rf_mean_mse:.4f} ± {rf_std_mse:.4f}")

# --- Linear Regression ---
lr_model = LinearRegression()
lr_mean_mse, lr_std_mse = evaluate_model(lr_model, X, y, weights)
print(f"Linear Regression MSE: {lr_mean_mse:.4f} ± {lr_std_mse:.4f}")
