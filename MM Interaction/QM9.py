import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import numpy as np
from utils import BasicAtomFeaturizer
from rdkit import Chem

MAX_DATA = 10000

ATOM_LIST = ["H", "B", "C", "N", "O", "F", "Na", "P", "S", "Cl", 
             "Ca", "I", "Br", "Se", "Mo", "Nb", "Ga", "Sb"]

def convert_qm9_to_custom_features():
    """Convert QM9 dataset using your custom featurizer"""
    
    # Load original QM9 dataset
    dataset = QM9(root='./data')
    
    # Initialize your featurizer
    featurizer = BasicAtomFeaturizer()
    
    processed_data = []
    
    print(f"Processing {len(dataset)} molecules with custom featurizer...")
    print(f"Feature dimension will be: {len(ATOM_LIST) + 5}")
    
    for i, data in enumerate(dataset):
        
        if i >= MAX_DATA:
            break

        try:
            # Get SMILES from the QM9 data object
            smiles = data.smiles  # QM9 dataset includes SMILES strings
            
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Could not parse SMILES for molecule {i}: {smiles}")
                continue
            
            # Add hydrogens to match the original molecule structure
            mol = Chem.AddHs(mol)
            
            # Generate custom features for each atom
            atom_features = []
            for atom in mol.GetAtoms():
                features = featurizer.encode(atom)
                atom_features.append(features)
            
            if len(atom_features) == 0:
                continue
            
            # Get edge connectivity from original QM9 data or reconstruct
            edge_index = data.edge_index
            
            # Create new data object with custom features
            new_data = Data(
                x=torch.tensor(atom_features, dtype=torch.float),
                edge_index=edge_index,
                y=data.y,  # Keep original targets
                smiles=smiles,
                name=data.name if hasattr(data, 'name') else f'mol_{i}'
            )
            
            processed_data.append(new_data)
            
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            continue
        
        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} molecules, successful: {len(processed_data)}")
    
    print(f"Successfully processed {len(processed_data)} molecules")
    if len(processed_data) > 0:
        print(f"Feature dimension: {processed_data[0].x.shape[1]}")
        
    return processed_data

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
        
        # Global mean and max pooling for graph-level representation
        if batch is not None:
            pooled_mean = global_mean_pool(x, batch)
            pooled_max = global_max_pool(x, batch)
        else:
            batch_zeros = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pooled_mean = global_mean_pool(x, batch_zeros)
            pooled_max = global_max_pool(x, batch_zeros)
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)

        # print(f"Pooled shape: {pooled.shape}")

        return pooled
    

class GCNRegressor(nn.Module):
    def __init__(self, node_dim, hidden_dim=32, gnn_out_dim=64, mlp_hidden=64, 
                 frozen = False, pretrained_encoder_path=None):
        super().__init__()
        
        # Initialize encoders
        self.encoder1 = GCNEncoder(node_dim, hidden_dim, gnn_out_dim, num_layers=3)
        self.encoder2 = GCNEncoder(node_dim, hidden_dim, gnn_out_dim, num_layers=3)
        
        # Load pretrained weights if provided
        if pretrained_encoder_path:
            checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
            
            # Extract the actual encoder state dict from the checkpoint
            encoder_state_dict = checkpoint['encoder_state_dict']
            
            # Load into both encoders
            self.encoder1.load_state_dict(encoder_state_dict)
            self.encoder2.load_state_dict(encoder_state_dict)
            
            # Verify dimensions match
            expected_dim = checkpoint.get('node_dim', node_dim)
            if expected_dim != node_dim:
                raise ValueError(f"Node dimension mismatch: expected {expected_dim}, got {node_dim}")
            
            print("Loaded pretrained encoder successfully!")
            
            # Freeze the encoder parameters

            if frozen:
                self.freeze_encoders()
            else:
                self.unfreeze_encoders()

        else:
            print("No pretrained encoder path provided, training from scratch.")

        # MLP for final prediction (this will be trainable)
        # Each encoder outputs [batch_size, 2 * gnn_out_dim] (mean + max pooling)
        # So concatenated: [batch_size, 4 * gnn_out_dim]
        self.mlp = nn.Sequential(
            nn.Linear(4 * gnn_out_dim, mlp_hidden),  # 2 encoders concatenated
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        
    
    def freeze_encoders(self):
        """Freeze encoder parameters to prevent updates during training"""
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        
        print("Encoders frozen - only MLP will be trained")
    
    def unfreeze_encoders(self):
        """Unfreeze encoders if you want to fine-tune later"""
        for param in self.encoder1.parameters():
            param.requires_grad = True
        for param in self.encoder2.parameters():
            param.requires_grad = True
        
        print("Encoders unfrozen")
    
    def forward(self, data):

        # print("Inside forward pass")

        h1 = self.encoder1(data.x1, data.edge_index1)
        h2 = self.encoder2(data.x2, data.edge_index2)

        # print(f"h1 shape: {h1.shape}, h2 shape: {h2.shape}")

        h = torch.cat([h1, h2], dim=-1)
        out = self.mlp(h)
        return out.view(-1)

# Training function for transfer learning


class QM9Pretrainer(nn.Module):
    def __init__(self, node_dim, hidden_dim=32, gnn_out_dim=64):
        super().__init__()
        self.encoder = GCNEncoder(node_dim, hidden_dim, gnn_out_dim, num_layers=3)
        self.predictor = nn.Sequential(
            nn.Linear(2*gnn_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 19)
        )
    
    def forward(self, data):
        x = self.encoder(data.x, data.edge_index, data.batch)
        return self.predictor(x)

def prepare_qm9_custom_data():
    """Prepare QM9 data with your custom featurizer"""
    
    # Convert QM9 to use custom features
    processed_data = convert_qm9_to_custom_features()
    
    if len(processed_data) == 0:
        raise ValueError("No molecules were successfully processed!")
    
    # Use a subset for faster training (adjust as needed)
    processed_data = processed_data[:10000]  # Use first 10k for demo
        
    # Select one target property - be safe about indexing
    for data in processed_data:
        if data.y.dim() > 0 and data.y.shape[0] > 2:
            # HOMO energy is at index 2 if we have enough properties
            data.y = data.y[2].unsqueeze(0)
        elif data.y.dim() > 0 and data.y.shape[0] > 0:
            # Use the first available property if index 2 doesn't exist
            data.y = data.y[0].unsqueeze(0)
        else:
            # If y is scalar or empty, just use it as is
            data.y = data.y.unsqueeze(0) if data.y.dim() == 0 else data.y

    # Debug: Check the target tensor shape before processing
    print(f"Sample target tensor shape: {processed_data[0].y.shape}")
    print(f"Sample target tensor: {processed_data[0].y}")
    
    # Split dataset
    train_size = int(0.8 * len(processed_data))
    val_size = int(0.1 * len(processed_data))
    
    train_dataset = processed_data[:train_size]
    val_dataset = processed_data[train_size:train_size + val_size]
    
    # Normalize target values
    train_targets = torch.cat([data.y for data in train_dataset])
    train_mean = train_targets.mean()
    train_std = train_targets.std()
    
    for data in processed_data:
        data.y = (data.y - train_mean) / train_std
    
    return train_dataset, val_dataset, train_mean, train_std

def pretrain_encoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data with custom features
    train_dataset, val_dataset, train_mean, train_std = prepare_qm9_custom_data()
    
    # Get node dimension from your custom featurizer
    node_dim = len(ATOM_LIST) + 5  # one-hot + 5 scalar features
    
    print(f"Using custom features with dimension: {node_dim}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = QM9Pretrainer(node_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    print(f"Training on device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Training loop
    for epoch in range(50):  # increased a bit for validation usefulness
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch).squeeze()
            loss = criterion(pred, batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Train Loss: {loss.item():.4f}')
        
        avg_train_loss = total_loss / len(train_loader)
        
        # ---- Validation step ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch).squeeze()
                loss = criterion(pred, batch.y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
    
    # Save the pretrained encoder with metadata
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'node_dim': node_dim,
        'hidden_dim': 32,
        'gnn_out_dim': 64,
        'featurizer_info': {
            'atom_list': ATOM_LIST,
            'feature_dim': node_dim
        }
    }, './models/pretrained_encoder_custom.pth')
    
    print("Pretrained encoder with custom features saved!")
    return model.encoder

if __name__ == "__main__":
    # This will now use your custom featurizer consistently
    pretrained_encoder = pretrain_encoder()
    
    # Later, when using with your paired molecules:
    node_dim = len(ATOM_LIST) + 5  # Same dimension as your custom featurizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device:", device)

    model = GCNRegressor(
        node_dim=node_dim,
        pretrained_encoder_path='models/pretrained_encoder_custom.pth'  # Path to your pretrained encoder
    ).to(device)
    
    print("Model ready for transfer learning with custom features!")