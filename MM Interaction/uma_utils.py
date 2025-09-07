# uma_utils.py
# Requires: pip install fairchem-core ase rdkit
# Also: request access to facebook/UMA on HuggingFace and `huggingface-cli login` (or set HF_TOKEN env var)
#
# Docs / references:
# - fairchem quickstart / UMA quickstart (FAIR-Chem docs). :contentReference[oaicite:1]{index=1}
import torch
from ase import Atoms
from ase.data import chemical_symbols

# Allow old-style slices (needed for FairChem checkpoints)
torch.serialization.add_safe_globals([slice])

# Now load normally via FairChem
from fairchem.core.calculate import pretrained_mlip
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")


import os
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional

# RDKit -> ASE helper
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

# FAIR-Chem UMA API
# Make sure fairchem-core is installed in the environment:
# pip install fairchem-core
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import os

from fairchem.core import pretrained_mlip
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")


logger = logging.getLogger(__name__)


def rdkit_to_ase(mol: Chem.Mol, charge: int = 0, spin: int = 0, add_hs: bool = True) -> Atoms:
    """
    Convert RDKit Mol -> ase.Atoms with a 3D conformer.
    Ensures a conformer exists (Embed + UFF optimize if needed).
    """
    if mol is None:
        raise ValueError("mol is None")

    # optionally add hydrogens
    if add_hs:
        mol = Chem.AddHs(mol)

    # ensure conformer
    if mol.GetNumConformers() == 0:
        # Use ETKDG embedding - good default
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            # optimization may fail for weird molecules; continue anyway
            pass

    conf = mol.GetConformer()
    positions = []
    symbols = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append((float(pos.x), float(pos.y), float(pos.z)))
        symbols.append(atom.GetSymbol())

    atoms = Atoms(symbols, positions=positions)
    # store charge / spin as metadata that FAIR-Chem / ASE expect
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    return atoms

import torch
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from fairchem.core import pretrained_mlip
# from fairchem.core.units.mlip_unit import infer_embeddings  # correct function to get latent features

import torch
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
# import pretrained_mlip


class UMAEncoder(torch.nn.Module):
    def __init__(self, model_name="uma-s-1p1", device=None, debug=False):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.predictor = pretrained_mlip.get_predict_unit(model_name, device=self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # Convert atomic numbers
        zs = (
            x.squeeze(-1).cpu().long().numpy()
            if x.dim() == 2 and x.size(-1) == 1
            else x.cpu().long().numpy()
        )
        pos_np = pos.cpu().numpy().astype(np.float64)
        symbols = [chemical_symbols[int(z)] for z in zs]

        atoms = Atoms(symbols, positions=pos_np)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 0

        # Run inference
        results = pretrained_mlip.infer(self.predictor, atoms)

        # Convert everything to torch tensors
        energy = torch.tensor(results["energy"], dtype=torch.float32, device=self.device).view(1, -1)
        forces = torch.tensor(results["forces"], dtype=torch.float32, device=self.device).flatten().view(1, -1)
        stress = torch.tensor(results.get("stress", np.zeros((3, 3))), dtype=torch.float32, device=self.device).flatten().view(1, -1)
        embeddings = torch.tensor(results["embeddings"], dtype=torch.float32, device=self.device).flatten().view(1, -1)

        # Concatenate into a single tensor
        out = torch.cat([energy, forces, stress, embeddings], dim=1)

        if self.debug:
            print(f"Final tensor shape: {out.shape}")

        return out

# --------------------
# Example usage snippet
# --------------------
if __name__ == "__main__":
    # quick local test: build small RDKit molecule and encode
    from rdkit.Chem import MolFromSmiles
    # set HF_TOKEN as env var or `huggingface-cli login` must be done beforehand
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)

    enc = UMAEncoder(uma_model_name="uma-s-1", task_name="omol", device=device, freeze=True, debug=True)

    # SMILES -> RDKit Mol -> coords
    smi = "CCCO"  # ethanol
    m = Chem.AddHs(MolFromSmiles(smi))
    if m.GetNumConformers() == 0:
        AllChem.EmbedMolecule(m, AllChem.ETKDG())
        try:
            AllChem.UFFOptimizeMolecule(m)
        except Exception:
            pass

    zs = torch.tensor([a.GetAtomicNum() for a in m.GetAtoms()], dtype=torch.long).unsqueeze(-1)  # [N,1]
    conf = m.GetConformer()
    pos = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())], dtype=torch.float32)

    emb = enc(zs, pos)
    print("embedding shape:", emb.shape)
    print("embedding:", emb.cpu().numpy())
