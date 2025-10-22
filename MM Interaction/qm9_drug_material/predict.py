import os
import pandas as pd
import torch
from rdkit import Chem
from utils import graph_from_molecule, PairedData, add_loops_to_data, load_mol_from_cif_or_sdf
from tune_decoder import DrugMaterialRegressor
from qm9_encoder import ATOM_LIST, NODE_DIM  # Import consistent dimensions

def load_best_parameters(results_file='results/decoder_tuning.csv'):
    """Load best parameters from tuning results"""
    df = pd.read_csv(results_file)

    best_params = df.loc[df['avg_rmse'].idxmin()].to_dict()
    print("Best decoder parameters found:", best_params)
    
    return best_params

def train_final_model(dataset, params, encoder_path, device):
    """Train final model with best parameters"""
    model = DrugMaterialRegressor(
    node_dim=int(NODE_DIM),
    hidden_dim=int(params['hidden_dim']),
    gnn_out_dim=int(params['gnn_out_dim']),
    mlp_hidden=int(params['mlp_hidden']),
    pretrained_encoder_path=encoder_path
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.MSELoss()
    
    # Training loop
    num_epochs = 200  # More epochs for final training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for data in dataset:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            # Ensure y is a tensor
            if isinstance(data.y, float):
                y = torch.tensor([data.y], dtype=pred.dtype, device=pred.device)
            else:
                y = data.y.to(device)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(dataset):.4f}")
    
    return model

def predict_interactions(model, material_file, sdf_file, output_file, device):
    """Generate predictions for drug-material pairs"""
    # Load material
    material = load_mol_from_cif_or_sdf(material_file)
    if material is None:
        raise ValueError(f"Could not load material from {material_file}")
    
    mat_graph = add_loops_to_data(graph_from_molecule(material))
    
    # Load drugs
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    results = []
    
    model.eval()
    with torch.no_grad():
        for i, drug in enumerate(suppl):
            if drug is None:
                continue
            
            try:
                # Get drug name
                name = drug.GetProp("_Name") if drug.HasProp("_Name") else f"drug_{i}"
                
                # Convert to graph
                drug_graph = add_loops_to_data(graph_from_molecule(drug))
                
                # Create paired data
                paired = PairedData(mat_graph, drug_graph, y=0.0, weight=1.0).to(device)
                
                # Predict
                pred = model(paired).item()
                
                results.append({
                    'drug_name': name,
                    'prediction': pred,
                    'num_atoms': drug.GetNumAtoms()
                })
                
            except Exception as e:
                print(f"Error processing drug {i}: {e}")
                continue
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} predictions to {output_file}")
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, default="models/best_qm9_encoder.pth")
    parser.add_argument("--params_file", type=str, default="results/decoder_tuning.csv")
    parser.add_argument("--material", type=str, default="CIF_files/Graphsene.cif")
    parser.add_argument("--sdf", type=str, default="20220301-L1300-FDA-approved-Drug-Library.sdf")
    parser.add_argument("--output", type=str, default="new_predictions.csv")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset for training
    from utils import build_pairs_from_csv
    dataset = build_pairs_from_csv("trial_data.csv", "CIF_files", "SDF_files")
    
    # Load best parameters and train model
    best_params = load_best_parameters(args.params_file)
    model = train_final_model(dataset, best_params, args.encoder_path, device)
    
    # Save trained model
    torch.save(model.state_dict(), "models/final_model.pth")
    
    # Generate predictions
    predictions = predict_interactions(
        model,
        args.material,
        args.sdf,
        args.output,
        device
    )
    
    # Print top predictions
    print("\nTop 10 predictions:")
    print(predictions.nlargest(10, 'prediction'))