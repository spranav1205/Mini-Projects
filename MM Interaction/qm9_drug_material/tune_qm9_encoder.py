import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from qm9_encoder import (
    process_qm9_dataset, 
    QM9Pretrainer,
    NODE_DIM
)
import itertools
import pandas as pd
import os

def grid_search_qm9(epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process dataset
    dataset = process_qm9_dataset()

    # Stack all targets to compute mean and std for each property
    all_targets = torch.cat([d.y.view(1, -1) for d in dataset], dim=0)  # [N, 19]
    target_mean = all_targets.mean(dim=0)
    target_std = all_targets.std(dim=0)

    # Normalize targets in dataset
    for d in dataset:
        d.y = (d.y - target_mean) / (target_std + 1e-8)

    # Split dataset into train/val
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # Define parameter grid
    param_grid = {
        'hidden_dim': [64, 128],
        'gnn_out_dim': [64, 128],
        'num_layers': [3, 4],
        'lr': [1e-3, 1e-4, 5e-4]
    }
    
    # Create all combinations
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Results storage
    results = []
    
    # Create data loaders once
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    for exp_idx, params in enumerate(experiments):
        print(f"\nExperiment {exp_idx + 1}/{len(experiments)}")
        print(f"Parameters: {params}")

        # Initialize model with current parameters
        model = QM9Pretrainer(
            node_dim=NODE_DIM,
            hidden_dim=params['hidden_dim'],
            gnn_out_dim=params['gnn_out_dim']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                # Normalize predictions for each property (already normalized targets)
                loss = criterion(pred, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    val_loss += criterion(pred, batch.y).item()

            train_loss = total_loss / len(train_loader)
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # Save best model for this parameter set
                checkpoint = {
                    'encoder_state_dict': model.encoder.state_dict(),
                    'params': params,
                    'val_loss': best_val_loss,
                    'epoch': epoch,
                    'target_mean': target_mean,
                    'target_std': target_std
                }
                os.makedirs('models', exist_ok=True)
                torch.save(checkpoint, f'models/qm9_encoder_exp{exp_idx}.pth')

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        results.append({
            **params,
            'val_loss': best_val_loss,
            'best_epoch': best_epoch
        })
        
        # Save progress after each experiment
        df = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        df.to_csv('results/qm9_encoder_tuning.csv', index=False)
        
        print(f"\nExperiment {exp_idx + 1} validation loss: {best_val_loss:.4f}")
    
    return results

if __name__ == "__main__":
    results = grid_search_qm9()
    
    # Print best parameters
    df = pd.DataFrame(results)
    best_idx = df['val_loss'].argmin()
    best_params = df.iloc[best_idx]
    
    print("\nBest parameters:")
    for param, value in best_params.items():
        if param not in ['val_loss', 'final_epoch']:
            print(f"{param}: {value}")
    
    # Copy best model to final location
    best_exp = df['val_loss'].argmin()
    os.replace(
        f'models/qm9_encoder_exp{best_exp}.pth',
        'models/best_qm9_encoder.pth'
    )