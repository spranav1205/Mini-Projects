import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import os
from rdkit import Chem
from utils import BasicAtomFeaturizer

# List of atoms for featurization - exported for use in other modules
ATOM_LIST = ["H", "B", "C", "N", "O", "F", "Na", "P", "S", "Cl", "Ca", "I", "Br", "Se", "Mo", "Nb", "Ga", "Sb"]
NODE_DIM = len(ATOM_LIST) + 5

class GCNEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
    def forward(self, x, edge_index, batch=None):
        for conv in self.convs[:-1]:
            x = torch.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        if batch is not None:
            pooled_mean = global_mean_pool(x, batch)
            pooled_max = global_max_pool(x, batch)
        else:
            batch_zeros = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pooled_mean = global_mean_pool(x, batch_zeros)
            pooled_max = global_max_pool(x, batch_zeros)
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return pooled

class QM9Pretrainer(nn.Module):
    def __init__(self, node_dim, hidden_dim=32, gnn_out_dim=64):
        super().__init__()
        self.encoder = GCNEncoder(node_dim, hidden_dim, gnn_out_dim)
        self.predictor = nn.Sequential(
            nn.Linear(2*gnn_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 19)
        )
    def forward(self, data):
        x = self.encoder(data.x, data.edge_index, data.batch)
        return self.predictor(x)

def process_qm9_dataset(max_molecules=10000):
    dataset = QM9(root='./data')
    featurizer = BasicAtomFeaturizer()
    processed_data = []
    for i, data in enumerate(dataset):
        if i >= max_molecules:
            break
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        atom_features = []
        for atom in mol.GetAtoms():
            features = featurizer.encode(atom)
            atom_features.append(features)
        if len(atom_features) == 0:
            continue
        new_data = Data(
            x=torch.tensor(atom_features, dtype=torch.float),
            edge_index=data.edge_index,
            y=data.y,
            smiles=smiles
        )
        processed_data.append(new_data)
    return processed_data

def train_qm9_encoder(batch_size=32, epochs=50, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = process_qm9_dataset()
    # Stack all targets to compute mean and std for each property
    all_targets = torch.cat([d.y.view(1, -1) for d in dataset], dim=0)  # [N, 19]
    target_mean = all_targets.mean(dim=0)
    target_std = all_targets.std(dim=0)
    # Normalize targets in dataset
    for d in dataset:
        d.y = (d.y - target_mean) / (target_std + 1e-8)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    model = QM9Pretrainer(NODE_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss += criterion(pred, batch.y).item()
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            os.makedirs('models', exist_ok=True)
            torch.save({
                'encoder_state_dict': model.encoder.state_dict(),
                'node_dim': NODE_DIM,
                'hidden_dim': 32,
                'gnn_out_dim': 64,
                'featurizer_info': {
                    'atom_list': ATOM_LIST,
                    'feature_dim': NODE_DIM
                },
                'target_mean': target_mean,
                'target_std': target_std,
                'best_epoch': best_epoch,
                'val_loss': best_val_loss
            }, 'models/best_qm9_encoder.pth')
    print(f'Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}')
    return model.encoder

if __name__ == "__main__":
    encoder = train_qm9_encoder()
    print("QM9 encoder training completed!")
