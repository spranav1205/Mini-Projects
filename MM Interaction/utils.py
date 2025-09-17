import os
import logging
import warnings
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    global_mean_pool as gap,
    global_max_pool as gmp,
)

# -------------------------------
# Logging
# -------------------------------
logger = logging.getLogger(__name__)

# -------------------------------
# Featurizers
# -------------------------------
from rdkit.Chem.rdchem import BondType

class Featurizer:
    def __init__(self, allowable_sets, direct_features=None):
        self.dim = 0
        self.features_mapping = {}
        self.direct_features = direct_features if direct_features else {}

        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(
                zip(s, range(self.dim, len(s) + self.dim))
            )
            self.dim += len(s)

        for k in self.direct_features.keys():
            self.features_mapping[k] = self.dim
            self.dim += 1

    def encode(self, inputs):
        output = np.zeros((self.dim,), dtype=float)
        for name_feature, feature_mapping in self.features_mapping.items():
            if name_feature in self.direct_features:
                continue
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0

        for name_feature, index in self.direct_features.items():
            feature = getattr(self, name_feature)(inputs)
            output[index] = float(feature)

        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets, direct_features):
        super().__init__(allowable_sets, direct_features)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs() if atom.HasProp("_TotalNumHs") else 0

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()

    def is_aromatic(self, atom):
        return float(atom.GetIsAromatic())

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def formal_charge(self, atom):
        return atom.GetFormalCharge()

    def atomic_number(self, atom):
        return atom.GetAtomicNum()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets, direct_features):
        super().__init__(allowable_sets, direct_features)

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return float(bond.GetIsConjugated())

    def bond_order(self, bond):
        bond_orders = {
            BondType.SINGLE: 1.0,
            BondType.DOUBLE: 2.0,
            BondType.TRIPLE: 3.0,
            BondType.AROMATIC: 1.5,
        }
        return bond_orders.get(bond.GetBondType(), 0.0)


# Expanded features (kept simple & relevant)
atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F","Ga", "H", "I", "N", "Na", "O", "P", "S","Sb","Se", "Mo", "Nb"},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    },
    direct_features={
        "formal_charge": None,
        "atomic_number": None,
        "is_aromatic": None,
        "n_hydrogens": None,
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {1.0, 0.0},
    },
    direct_features={
        "bond_order": None,
    }
)

DEFAULT_NODE_DIM = 23  # base expected node dim
DEFAULT_EDGE_DIM = bond_featurizer.dim

ATOM_LIST = ["H", "B", "C", "N", "O", "F", "Na", "P", "S", "Cl", 
             "Ca", "I", "Br", "Se", "Mo", "Nb", "Ga", "Sb"]

ATOMIC_MASS = {
    "H": 1.008, "B": 10.81, "C": 12.01, "N": 14.01, "O": 16.00,
    "F": 18.998, "Na": 22.99, "P": 30.97, "S": 32.06, "Cl": 35.45,
    "Ca": 40.08, "I": 126.90, "Br": 79.90, "Se": 78.96, "Mo": 95.95,
    "Nb": 92.91, "Ga": 69.72, "Sb": 121.76
}

VALENCY = {
    "H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1,
    "Na": 1, "P": 3, "S": 2, "Cl": 1, "Ca": 2,
    "I": 1, "Br": 1, "Se": 2, "Mo": 6, "Nb": 5,
    "Ga": 3, "Sb": 5
}

ELECTRONEGATIVITY = {
    "H": 2.20, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ca": 1.00,
    "I": 2.66, "Br": 2.96, "Se": 2.55, "Mo": 2.16, "Nb": 1.6,
    "Ga": 1.81, "Sb": 2.05
}

class BasicAtomFeaturizer:
    def __init__(self, atom_list=ATOM_LIST):
        self.atom_list = atom_list
        self.atom_index = {a: i for i, a in enumerate(atom_list)}

    def encode(self, atom, conf=None):
        symbol = atom.GetSymbol()

        # One-hot atom identity
        one_hot = np.zeros(len(self.atom_list))
        if symbol in self.atom_index:
            one_hot[self.atom_index[symbol]] = 1.0

        # Safe scalar lookups
        mass = ATOMIC_MASS.get(symbol, 0.0)
        valency = VALENCY.get(symbol, 0.0)
        eneg = ELECTRONEGATIVITY.get(symbol, 0.0)

        try:
            degree = atom.GetDegree()
        except Exception:
            degree = 0

        try:
            formal_charge = atom.GetFormalCharge()
        except Exception:
            formal_charge = 0

        # # Coordinates
        # coords = np.zeros(3)
        # if conf is not None:
        #     try:
        #         pos = conf.GetAtomPosition(atom.GetIdx())
        #         coords = np.array([pos.x, pos.y, pos.z])
        #     except Exception:
        #         pass

        return np.concatenate([
            one_hot,
            # [mass, valency, eneg, degree, formal_charge], coords])
            [mass, valency, eneg, degree, formal_charge]])


basic_featurizer = BasicAtomFeaturizer()

# -------------------------------
# Molecule to graph
# -------------------------------
def molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_from_molecule(molecule):
    if molecule is None:
        raise ValueError("graph_from_molecule received None molecule")

    atom_features = []
    bond_features = []
    edge_index = []

    # --- New Atom Features ---
    for atom in molecule.GetAtoms():
        vec = basic_featurizer.encode(atom)
        atom_features.append(vec)

    # --- Bond Features ---
    for bond in molecule.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([s, e])
        edge_index.append([e, s])
        bf = bond_featurizer.encode(bond)
        bond_features.append(bf)
        bond_features.append(bf)

    # --- Convert to tensors ---
    x = torch.tensor(atom_features, dtype=torch.float)
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_featurizer.dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(bond_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def add_loops_to_data(data: Data) -> Data:
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    data.edge_index = edge_index
    return data


class PairedData(Data):
    def __init__(self, data1, data2, y=0.0, weight=1.0):
        super().__init__()
        self.x1 = data1.x
        self.edge_index1 = data1.edge_index
        self.edge_attr1 = data1.edge_attr

        self.x2 = data2.x
        self.edge_index2 = data2.edge_index
        self.edge_attr2 = data2.edge_attr

        self.y = float(y) if y is not None else 0.0
        self.w = torch.tensor([weight], dtype=torch.float)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index1":
            return self.x1.shape[0] if self.x1 is not None else 0
        if key == "edge_index2":
            return self.x2.shape[0] if self.x2 is not None else 0
        return super().__inc__(key, value, *args, **kwargs)


# -------------------------------
# Models
# -------------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super().__init__()
        assert num_layers >= 3
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = torch.cat([gap(x, batch=None), gmp(x, batch=None)], dim=1)
        return pooled


class GCNRegressor(nn.Module):
    def __init__(self, node_dim, hidden_dim=32, gnn_out_dim=64, mlp_hidden=64):
        super().__init__()
        self.encoder1 = GCNEncoder(node_dim, hidden_dim, gnn_out_dim, num_layers=3)
        self.encoder2 = GCNEncoder(node_dim, hidden_dim, gnn_out_dim, num_layers=3)

        self.mlp = nn.Sequential(
            nn.Linear(4 * gnn_out_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, data):
        h1 = self.encoder1(data.x1, data.edge_index1)
        h2 = self.encoder2(data.x2, data.edge_index2)
        h = torch.cat([h1, h2], dim=-1)
        out = self.mlp(h)
        return out.view(-1)


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=3, dropout=0.4):
        super().__init__()
        assert num_layers >= 3
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATRegressor(nn.Module):
    def __init__(self, node_dim, hidden_dim=32, gnn_out_dim=64, heads=4):
        super().__init__()
        self.encoder1 = GATEncoder(node_dim, hidden_dim, gnn_out_dim, heads=heads, num_layers=3)
        self.encoder2 = GATEncoder(node_dim, hidden_dim, gnn_out_dim, heads=heads, num_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(4 * gnn_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        x1 = self.encoder1(data.x1, data.edge_index1)
        h1 = torch.cat([gmp(x1, batch=None), gap(x1, batch=None)], dim=1)

        x2 = self.encoder2(data.x2, data.edge_index2)
        h2 = torch.cat([gmp(x2, batch=None), gap(x2, batch=None)], dim=1)

        h = torch.cat([h1, h2], dim=-1)
        return self.fc(h).squeeze(-1)


def build_allowed_mask(n1: int, n2: int, edge_index1: torch.Tensor, edge_index2: torch.Tensor, device=None):
    L = n1 + n2
    allowed = torch.zeros((L, L), dtype=torch.bool, device=device)

    A1 = torch.zeros((n1, n1), dtype=torch.bool, device=device)
    if edge_index1.numel() > 0:
        src, dst = edge_index1
        A1[dst, src] = True
        A1[src, dst] = True
    A1.fill_diagonal_(True)
    allowed[:n1, :n1] = A1

    A2 = torch.zeros((n2, n2), dtype=torch.bool, device=device)
    if edge_index2.numel() > 0:
        src2, dst2 = edge_index2
        A2[dst2, src2] = True
        A2[src2, dst2] = True
    A2.fill_diagonal_(True)
    allowed[n1:, n1:] = A2

    allowed[:n1, n1:] = True
    allowed[n1:, :n1] = True
    return allowed


class TransformerRegressor(nn.Module):
    def __init__(self, node_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(node_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, data):
        device = next(self.parameters()).device
        x1 = data.x1.to(device).float()
        x2 = data.x2.to(device).float()
        ei1 = data.edge_index1.to(device).long() if data.edge_index1 is not None else torch.empty((2,0), dtype=torch.long, device=device)
        ei2 = data.edge_index2.to(device).long() if data.edge_index2 is not None else torch.empty((2,0), dtype=torch.long, device=device)

        n1, n2 = x1.size(0), x2.size(0)
        h1 = self.in_proj(x1) + self.type_embed(torch.zeros(n1, dtype=torch.long, device=device))
        h2 = self.in_proj(x2) + self.type_embed(torch.ones(n2, dtype=torch.long, device=device))
        H = torch.cat([h1, h2], dim=0).unsqueeze(0)

        allowed = build_allowed_mask(n1, n2, ei1, ei2, device=device)
        attn_mask = (~allowed).float() * -1e9

        Z = self.encoder(H, mask=attn_mask)
        Z_nodes = Z[0]

        z1_mean = Z_nodes[:n1].mean(dim=0, keepdim=True)
        z1_max  = Z_nodes[:n1].max(dim=0, keepdim=True).values
        z2_mean = Z_nodes[n1:].mean(dim=0, keepdim=True)
        z2_max  = Z_nodes[n1:].max(dim=0, keepdim=True).values

        z1 = torch.cat([z1_mean, z1_max], dim=-1)
        z2 = torch.cat([z2_mean, z2_max], dim=-1)

        h = torch.cat([z1, z2], dim=-1)
        out = self.head(h)
        return out.view(-1)


# -------------------------------
# Train / Eval
# -------------------------------
def rmse_loss(pred, target):
    return torch.sqrt(nn.MSELoss()(pred, target))

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        target = torch.tensor([batch.y], dtype=torch.float, device=device)
        loss = rmse_loss(out, target)
        (loss * batch.w.to(device)).backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        target = torch.tensor([batch.y], dtype=torch.float, device=device)
        loss = rmse_loss(out, target)
        total += abs(loss.item())
    return total / max(1, len(loader))
