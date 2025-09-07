import os
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import numpy as np
from fairchem.core.calculate import pretrained_mlip
from ase import Atoms
from uma_utils import UMAEncoder  # your wrapper

logger = logging.getLogger("UMAEmbedding")
logging.basicConfig(level=logging.INFO)

CSV_PATH = "trial_dataCopy.csv"
CIF_DIR = "CIF_files/"
SDF_DIR = "SDF_files/"
SAVE_DIR = "embeddings_cache/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load UMA encoder once
encoder = UMAEncoder(uma_model_name="uma-s-1p1", device="cpu", freeze=True, debug=False)

def load_mol_from_cif_or_sdf(base_path, max_atoms=200):
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
    elif os.path.exists(sdf):
        suppl = Chem.SDMolSupplier(sdf, removeHs=False)
        candidates = [m for m in suppl if m is not None]
        mol = candidates[0] if candidates else None
    return mol

def embed_and_save(mol, base_name, prefix):
    out_path = os.path.join(SAVE_DIR, f"{prefix}_{base_name}.pt")
    if os.path.exists(out_path):
        return torch.load(out_path)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Prepare atomic numbers and positions
    conf = mol.GetConformer()
    x = torch.tensor([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())], dtype=torch.long)
    pos = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] 
                        for i in range(mol.GetNumAtoms())], dtype=torch.float32)
    
    try:
        emb = encoder(x, pos).detach()  # UMAEncoder expects x and pos
        torch.save(emb, out_path)
        return emb
    except Exception as e:
        logger.warning(f"Failed to encode {base_name}: {e}")
        return None

# Load CSV
df = pd.read_csv(CSV_PATH)

material_embeddings = []
drug_embeddings = []

# Materials
for i, mat in enumerate(df["material"]):
    mol = load_mol_from_cif_or_sdf(os.path.join(CIF_DIR, mat))
    emb = embed_and_save(mol, mat, "mat")
    material_embeddings.append(emb)
    if (i+1) % 10 == 0:
        logger.info(f"Processed {i+1}/{len(df)} materials")

# Drugs
for i, drug in enumerate(df["drug"]):
    mol = load_mol_from_cif_or_sdf(os.path.join(SDF_DIR, drug))
    emb = embed_and_save(mol, drug, "drug")
    drug_embeddings.append(emb)
    if (i+1) % 10 == 0:
        logger.info(f"Processed {i+1}/{len(df)} drugs")

# Save dataset
torch.save({
    "material_embeddings": material_embeddings,
    "drug_embeddings": drug_embeddings,
    "y": torch.tensor(df["y"].values, dtype=torch.float32)
}, os.path.join(SAVE_DIR, "dataset.pt"))

logger.info("Embeddings saved. You can now load `dataset.pt` for GPU training.")
