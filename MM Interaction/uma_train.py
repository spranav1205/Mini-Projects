import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import os

# --- Paths & device ---
SAVE_DIR = "embeddings_cache/"
DATA_PATH = os.path.join(SAVE_DIR, "dataset.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# --- Load cached embeddings ---
data = torch.load(DATA_PATH)
materials = data["material_embeddings"]
drugs = data["drug_embeddings"]
y = data["y"]

# --- Dataset class ---
class PairedEmbeddingDataset(Dataset):
    def __init__(self, mats, drugs, targets):
        self.mats = mats
        self.drugs = drugs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "mat": self.mats[idx],
            "drug": self.drugs[idx],
            "y": self.targets[idx]
        }

dataset = PairedEmbeddingDataset(materials, drugs, y)

# --- Simple MLP Regressor ---
class PairedRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mat_emb, drug_emb):
        x = torch.cat([mat_emb, drug_emb], dim=-1)
        return self.mlp(x).squeeze(-1)

# --- K-Fold Validation ---
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
input_dim = materials[0].shape[-1]

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n=== Fold {fold+1}/{k} ===")
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    model = PairedRegressor(input_dim=input_dim, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # --- Training loop ---
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            mat_emb = torch.stack(batch["mat"]).to(DEVICE)
            drug_emb = torch.stack(batch["drug"]).to(DEVICE)
            targets = batch["y"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(mat_emb, drug_emb)
            loss = torch.sqrt(criterion(preds, targets))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        avg_train_loss = total_loss / len(train_subset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mat_emb = torch.stack(batch["mat"]).to(DEVICE)
                drug_emb = torch.stack(batch["drug"]).to(DEVICE)
                targets = batch["y"].to(DEVICE)
                preds = model(mat_emb, drug_emb)
                val_loss += torch.sqrt(criterion(preds, targets)).item() * len(targets)

        avg_val_loss = val_loss / len(val_subset)
        print(f"Epoch {epoch+1}, Train RMSE: {avg_train_loss:.4f}, Val RMSE: {avg_val_loss:.4f}")
