import os
import logging
import itertools
import pandas as pd
import torch
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from utils import (
    graph_from_molecule, PairedData, add_loops_to_data,
    evaluate, train_one_epoch, DEFAULT_NODE_DIM
)
from qm9_encoder import GCNEncoder

class DrugMaterialRegressor(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim=32, gnn_out_dim=64, mlp_hidden=64, 
                 pretrained_encoder_path=None, lr=0.0, avg_rmse=0.0, frozen=True):
        super().__init__()
        
        # Initialize encoders
        self.encoder1 = GCNEncoder(node_dim, hidden_dim, gnn_out_dim)
        self.encoder2 = GCNEncoder(node_dim, hidden_dim, gnn_out_dim)
        
        if pretrained_encoder_path:
            checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
            encoder_state_dict = checkpoint['encoder_state_dict']
            
            # Load and freeze encoders
            self.encoder1.load_state_dict(encoder_state_dict)
            self.encoder2.load_state_dict(encoder_state_dict)

            if frozen:
            
                for param in self.encoder1.parameters():
                    param.requires_grad = False
                for param in self.encoder2.parameters():
                    param.requires_grad = False
                
                print("Loaded and frozen pretrained encoder")

            else :
            
                print("Loaded pretrained encoder without freezing")
        
        
        # MLP decoder
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * gnn_out_dim, mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden, 1)
        )
    
    def forward(self, data):
        h1 = self.encoder1(data.x1, data.edge_index1)
        h2 = self.encoder2(data.x2, data.edge_index2)
        h = torch.cat([h1, h2], dim=-1)
        return self.mlp(h).view(-1)

def run_decoder_grid_search(dataset, device, pretrained_encoder_path, epochs=100, k=5):
    results = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Parameter grid for decoder
    param_grid = {
        'hidden_dim': [32, 64],
        'gnn_out_dim': [64, 128],
        'mlp_hidden': [64, 128, 256],
        'lr': [1e-3, 5e-4, 1e-4]
    }
    
    keys, values = zip(*param_grid.items())
    
    for params in itertools.product(*values):
        params = dict(zip(keys, params))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            model = DrugMaterialRegressor(
                node_dim=DEFAULT_NODE_DIM,
                pretrained_encoder_path=pretrained_encoder_path,
                **params
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            
            best_val_score = float('inf')
            for epoch in range(epochs):
                train_loss = train_one_epoch(model, train_subset, optimizer, device)
                val_score = evaluate(model, val_subset, device)
                
                if val_score < best_val_score:
                    best_val_score = val_score
                
                if epoch % 10 == 0:
                    print(f"Fold {fold}, Epoch {epoch}: Train Loss = {train_loss:.4f}, Val RMSE = {val_score:.4f}")
            
            fold_scores.append(best_val_score)
        
        avg_rmse = sum(fold_scores) / len(fold_scores)
        results.append({**params, 'avg_rmse': avg_rmse})
        
        # Save progress
        df = pd.DataFrame(results)
        df.to_csv('results/decoder_tuning.csv', index=False)
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, default="models/best_qm9_encoder.pth")
    parser.add_argument("--csv", type=str, default="trial_data.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your dataset
    from utils import build_pairs_from_csv
    dataset = build_pairs_from_csv(args.csv, "CIF_files", "SDF_files")
    
    results = run_decoder_grid_search(
        dataset, 
        device,
        args.encoder_path,
        epochs=args.epochs,
        k=args.k
    )
    
    # Print best parameters
    df = pd.DataFrame(results)
    best_idx = df['avg_rmse'].argmin()
    best_params = df.iloc[best_idx]
    
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")