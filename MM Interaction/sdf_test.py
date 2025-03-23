from rdkit.Chem import Draw
from rdkit import Chem

supplier = Chem.SDMolSupplier(r"C:\Users\spran\Desktop\Mini-Projects\Drug Discovery Notebook\Basics\molecule.sdf")

for mol in supplier:
    image = Draw.MolToImage(mol)  # 2D visualization
    image.show()
