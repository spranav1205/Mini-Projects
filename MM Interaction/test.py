# requirements: rdkit, dgl, torch
# pip install rdkit-pypi dgl torch
# import os
# os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"

import dgl
print("DGL version:", dgl.__version__)

import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Convert RDKit molecule to DGL graph
def mol_to_dgl(mol):
    num_atoms = mol.GetNumAtoms()
    g = dgl.DGLGraph()
    g.add_nodes(num_atoms)

    # node features: atom type as one-hot
    atom_feats = []
    atom_types = ['H','C','N','O','F','P','S','Cl','Br','I']  # common atoms
    for atom in mol.GetAtoms():
        one_hot = [int(atom.GetSymbol() == a) for a in atom_types]
        atom_feats.append(one_hot)
    g.ndata['h'] = torch.tensor(atom_feats, dtype=torch.float32)

    # edges
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        g.add_edges(a1, a2)
        g.add_edges(a2, a1)  # undirected

    return g

# 2. Define a simple GCN for graph embedding
class GCNGraphEmbed(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)
        self.pool = dgl.nn.GlobalAveragePooling()

    def forward(self, g):
        h = g.ndata['h']
        h = torch.relu(self.conv1(g, h))
        h = torch.relu(self.conv2(g, h))
        hg = self.pool(g, h)
        return hg

# Example usage
smiles = "CCO"  # ethanol
mol = Chem.MolFromSmiles(smiles)
g = mol_to_dgl(mol)

model = GCNGraphEmbed(in_feats=10, hidden_feats=32, out_feats=128)  # embedding size 128
embedding = model(g)

print("Embedding shape:", embedding.shape)
print(embedding)
