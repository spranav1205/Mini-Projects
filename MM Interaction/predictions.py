import os
import pandas as pd
from rdkit import Chem
from utils import graph_from_molecule, PairedData, add_loops_to_data
from hyperparameter_tuning import load_mol_from_cif_or_sdf, build_pairs_from_csv

import os
import pandas as pd
from rdkit import Chem
from utils import graph_from_molecule, PairedData, add_loops_to_data
from hyperparameter_tuning import load_mol_from_cif_or_sdf

import torch
from torch_geometric.loader import DataLoader

# --- utils brings all helpers & models ---
from utils import (
    graph_from_molecule,
    PairedData,
    add_loops_to_data,
    GCNRegressor,
    GATRegressor,
    TransformerRegressor,
    DEFAULT_NODE_DIM,
    DEFAULT_EDGE_DIM,
    train_one_epoch,                 # NEW: now exists in utils
)

# -------------------------------
# CONFIG
# -------------------------------
num_epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define feature dimensions (were previously undefined)
node_dim = DEFAULT_NODE_DIM
edge_dim = DEFAULT_EDGE_DIM

paired_data_list = build_pairs_from_csv("./trial_data.csv", cif_dir="./CIF_files/", sdf_dir="./SDF_files/")
    

def train(name = "GCN"):
    # -------------------------------
    # Fresh model for final training
    # -------------------------------
    if name == "GCN":
        model = GCNRegressor(node_dim=node_dim, hidden_dim=64, gnn_out_dim=64).to(device)
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

        torch.save(model.state_dict(), f"./models/{name}_trained_model.pth")

    else:
        print("No training data found (paired_data_list is empty). Skipping training.")

    return model

def process_sdf_and_predict(sdf_file, material_file, model, csv_out="predictions.csv"):
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
    df = pd.DataFrame({
        "name": names,
        "num_atoms": atom_counts,
        "pred_energy": preds
    })
    df.to_csv(csv_out, index=False)
    print(f"Saved {len(df)} predictions to {csv_out}")

# --- Usage example ---
if __name__ == "__main__":
    name = "Transformer"  # or "GAT" or "Transformer"

    model = train(name)

    mat_file = "CIF_files/Graphsene"
    sdf_file = "Approveddrugslibrary.sdf"

    # model = GCNRegressor(node_dim=node_dim, hidden_dim=64, gnn_out_dim=128)  # Or load your trained model
    process_sdf_and_predict(sdf_file, mat_file, model, f"drug_predictions_{name}.csv")
