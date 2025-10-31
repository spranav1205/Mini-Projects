import os
import pandas as pd
import torch
import random
import numpy as np
from rdkit import Chem
from utils import graph_from_molecule, PairedData, add_loops_to_data, build_pairs_from_csv, load_mol_from_cif_or_sdf
from tune_decoder import DrugMaterialRegressor
from qm9_encoder import ATOM_LIST, NODE_DIM
from torch_geometric.data import DataLoader
import torch.nn as nn


# --- utils brings all helpers & models ---
from utils import (
    graph_from_molecule,
    PairedData,
    add_loops_to_data,
    # GCNRegressor,
    GATRegressor,
    TransformerRegressor,
    DEFAULT_NODE_DIM,
    DEFAULT_EDGE_DIM,
    train_one_epoch,                 # NEW: now exists in utils
    basic_featurizer
)

# -------------------------------
# CONFIG
# -------------------------------
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define feature dimensions (were previously undefined)
node_dim = DEFAULT_NODE_DIM
edge_dim = DEFAULT_EDGE_DIM

paired_data_list = build_pairs_from_csv("./trial_data.csv", cif_dir="./CIF_files/", sdf_dir="./SDF_files/")

print("Sample data point:",paired_data_list[0])

def train_with_encoder(train_loader, val_loader, frozen=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get node dimension
    node_dim = len(basic_featurizer.atom_list) + 5
    
    # Initialize model with pretrained encoder
    model = DrugMaterialRegressor(
        node_dim=node_dim,
        pretrained_encoder_path='models/best_qm9_encoder.pth',  # Path to your pretrained encoder
        frozen=frozen
    ).to(device)
    
    # Create optimizer - only for trainable parameters (MLP)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=0.0005)
    criterion = nn.MSELoss()
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):  # Fewer epochs since only MLP is training
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            loss = criterion(pred, batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')
    
    return model

def train(name = "GCN"):
    # -------------------------------
    # Fresh model for final training
    # -------------------------------
    if name == "GCN":
        model = DrugMaterialRegressor(
        node_dim=node_dim,
        pretrained_encoder_path='models/best_qm9_encoder.pth',
        frozen=False
        ).to(device)
    elif name == "GAT":
        model = GATRegressor(node_dim=node_dim, hidden_dim=64, gnn_out_dim=128).to(device)
    else:
        model = TransformerRegressor(node_dim=node_dim, d_model=64, nhead=8, num_layers=2, dim_feedforward=256).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # criterion = torch.nn.MSELoss()

    print(f"Training model on {len(paired_data_list)} samples...")

    # Guard: only train if we actually have data
    if len(paired_data_list) > 0:
        for epoch in range(0, num_epochs + 1):
            train_loss = train_one_epoch(model, paired_data_list, optimizer, device)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        # Save the trained model
        torch.save(model.state_dict(), f"./models/{name}_trained_model.pth")

    else:
        print("No training data found (paired_data_list is empty). Skipping training.")

    return model, train_loss

def process_sdf_and_predict(sdf_file, material_file, model, csv_out="predictions.csv", required_drugs=None):
    """
    sdf_file      : Path to drug SDF file
    material_file : Path to CIF or SDF file describing material
    model         : Trained GNN model (already loaded)
    csv_out       : Where to save predictions
    """
    # --- Load molecules from SDF ---
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    drugs = [mol for mol in suppl if mol is not None]
    if not drugs:
        raise ValueError(f"No valid molecules found in {sdf_file}")
    # Minimal filter for required drugs
    if required_drugs is not None:
        filtered_drugs = []
        for drug in drugs:
            if drug is None:
                continue
            try:
                if drug.HasProp("Name"):
                    name = drug.GetProp("Name")
                elif drug.HasProp("_Name"):
                    name = drug.GetProp("_Name")
                else:
                    name = None
            except UnicodeDecodeError:
                name = "decode_error"
            if name and any(req.lower() in name.lower() for req in required_drugs):
                filtered_drugs.append(drug)
        drugs = filtered_drugs
        print(f"Filtered to {len(drugs)} required drugs.")

    # --- Load & build material graph once ---
    material = load_mol_from_cif_or_sdf(material_file)
    if material is None:
        raise ValueError(f"Could not load material from {material_file}")

    mat_graph = add_loops_to_data(graph_from_molecule(material))
    print(f"Material graph built with {mat_graph.x.shape[0]} atoms")

    # --- Prepare for predictions ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds, names, atom_counts = [], [], []

    # --- Process & predict one drug at a time ---
    with torch.no_grad():
        for i, drug in enumerate(drugs):
            try:
                # Name handling
                if drug.HasProp("Name"):
                    name = drug.GetProp("Name")
                elif drug.HasProp("_Name"):
                    name = drug.GetProp("_Name")
                else:
                    name = f"drug_{i}"

                num_atoms = drug.GetNumAtoms()

                # Convert to graph
                drug_graph = add_loops_to_data(graph_from_molecule(drug))
                # drug_graph = graph_from_molecule(drug)  # No loops for drugs
                if drug_graph.x is None or drug_graph.x.size(0) == 0:
                    print(f"Skipping {name}: invalid drug graph")
                    continue

                # Build paired data (material + drug)
                paired = PairedData(mat_graph, drug_graph, y=0.0, weight=1.0).to(device)

                # Single forward pass
                out = model(paired)
                pred = out.view(-1).item()

                preds.append(pred)
                names.append(name)
                atom_counts.append(num_atoms)

                print(f"Predicted {name} (atoms={num_atoms}): {pred:.4f}")

            except Exception as e:
                print(f"Skipping drug {i} due to error: {e}")
                continue

    # --- Save predictions ---
    # Provide both legacy and normalized column names so downstream scripts can consume the output
    df = pd.DataFrame({
        "name": names,
        "drug_name": names,
        "num_atoms": atom_counts,
        "pred_energy": preds,
        "prediction": preds
    })
    df.to_csv(csv_out, index=False)
    print(f"Saved {len(df)} predictions to {csv_out}")

# --- Usage example ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/best_qm9_encoder.pth')
    parser.add_argument('--material', type=str, default='CIF_files/Graphsene')
    parser.add_argument('--sdf', type=str, default='20220301-L1300-FDA-approved-Drug-Library.sdf')
    parser.add_argument('--output', type=str, default='drug_predictions_without_data_GCN.csv')
    parser.add_argument('--required_drugs', type=str, default=None, help='Comma-separated list of drug names to predict')
    args = parser.parse_args()

    name = "GCN"
    model, final_loss = train(name)

    required_drugs = None
    if args.required_drugs:
        required_drugs = [d.strip() for d in args.required_drugs.split(",")]

    process_sdf_and_predict(args.sdf, args.material, model, args.output, required_drugs=required_drugs)

    print("Final Loss of model:", final_loss)
    print("Prediction process completed.")
