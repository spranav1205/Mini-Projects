import schnetpack as spk
# from schnetpack.atoms import AtomsData
from ase import Atoms
import torch

# Load pretrained SchNet model from QM9
model_path = spk.data.download_url(
    url="https://schnetpack.readthedocs.io/en/stable/_downloads/schnet_qm9_model.tar.gz",
    path="./pretrained_models"
)

# Load the model
model = spk.interfaces.AtomisticModel.load(model_path)
model.eval()  # set to evaluation mode

# Example molecule: water
atoms = Atoms(
    symbols=["O", "H", "H"],
    positions=[[0.0, 0.0, 0.0],
               [0.0, 0.757, 0.586],
               [0.0, -0.757, 0.586]]
)

# Convert ASE Atoms to SchNetPack input
z = torch.tensor(atoms.numbers, dtype=torch.long).unsqueeze(0)  # [1, n_atoms]
pos = torch.tensor(atoms.positions, dtype=torch.float32).unsqueeze(0)  # [1, n_atoms, 3]

# Forward pass to get embeddings
with torch.no_grad():
    # SchNet can return atom-wise representations
    atomwise_embeddings = model.representation(z, pos)  # [1, n_atoms, embedding_dim]

    # Pool to get a single molecular embedding
    mol_embedding = atomwise_embeddings.mean(dim=1)  # [1, embedding_dim]

print("Molecular embedding shape:", mol_embedding.shape)
