import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CutOffDictNN
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cif_file = r"C:\Users\spran\Desktop\cif\Irida_graphene.cif"
structure = Structure.from_file(cif_file)
#structure = structure.get_primitive_structure()  
'''
mol = Chem.RWMol()  # Create an editable molecule in RDKit

for site in structure:
    element = site.specie.symbol  # Get atomic symbol (e.g., "C", "O")
    atom = Chem.Atom(element)  # Create an RDKit atom
    mol.AddAtom(atom)  # Add it to the molecule

print(structure[0].coords)

structure.remove_sites([i for i, site in enumerate(structure) if site.is_periodic])

cutoff = 1.5  # Example: typical bond length for C-C or C-H


for i in range(len(structure)):
    for j in range(i + 1, len(structure)):
        dist = structure.get_distance(i, j)
        if dist < cutoff:  # Only consider distances within the cutoff
            print(f"Bond between atom {i} and atom {j}: {dist:.3f} Å")
            if mol.GetBondBetweenAtoms(i, j) is None:  
                mol.AddBond(i, j, Chem.BondType.SINGLE)  # Only add if not present'
         


neighbors = structure.get_neighbors(structure, r=cutoff)

for site, neighbor_list in zip(structure, neighbors):
    for neighbor in neighbor_list:
        i = structure.index(site)
        j = structure.index(neighbor.site)

        # Ensure atoms are from the same unit cell
        if neighbor.image == (0, 0, 0):  
            dist = neighbor.nn_distance
            print(f"Bond between atom {i} and atom {j}: {dist:.3f} Å")
            if mol.GetBondBetweenAtoms(i, j) is None:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

from rdkit.Chem import AllChem

conf = Chem.Conformer(mol.GetNumAtoms())  # Create a conformer

for idx, site in enumerate(structure):
    coord = site.coords  # Cartesian coordinates (x, y, z)
    conf.SetAtomPosition(idx, coord)  # Assign position

mol.AddConformer(conf)  # Add conformer to molecule
'''

def cif_to_mol(cif_file, cutoff):

    if not cif_file.endswith('.cif'):
        raise ValueError(f"The provided file '{cif_file}' is not a CIF file. Please provide a valid CIF file.")
    
    structure = Structure.from_file(cif_file)

    mol = Chem.RWMol()  # Create an editable molecule in RDKit

    for site in structure:
        element = site.specie.symbol 
        atom = Chem.Atom(element) 
        mol.AddAtom(atom) 

    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            dist = structure.get_distance(i, j)
            if dist < cutoff:  
                if mol.GetBondBetweenAtoms(i, j) is None:  
                    mol.AddBond(i, j, Chem.BondType.SINGLE) 

    conf = Chem.Conformer(mol.GetNumAtoms()) 

    for idx, site in enumerate(structure):
        coord = site.coords 
        conf.SetAtomPosition(idx, coord) 

    mol.AddConformer(conf) 

    return mol

mol = cif_to_mol(cif_file, 1.5)

from rdkit.Chem import Draw

image = Draw.MolToImage(mol)  # 2D visualization
image.show()
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CutOffDictNN
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cif_file = r"C:\Users\spran\Desktop\cif\Irida_graphene.cif"
structure = Structure.from_file(cif_file)
#structure = structure.get_primitive_structure()  
'''
mol = Chem.RWMol()  # Create an editable molecule in RDKit

for site in structure:
    element = site.specie.symbol  # Get atomic symbol (e.g., "C", "O")
    atom = Chem.Atom(element)  # Create an RDKit atom
    mol.AddAtom(atom)  # Add it to the molecule

print(structure[0].coords)

structure.remove_sites([i for i, site in enumerate(structure) if site.is_periodic])

cutoff = 1.5  # Example: typical bond length for C-C or C-H


for i in range(len(structure)):
    for j in range(i + 1, len(structure)):
        dist = structure.get_distance(i, j)
        if dist < cutoff:  # Only consider distances within the cutoff
            print(f"Bond between atom {i} and atom {j}: {dist:.3f} Å")
            if mol.GetBondBetweenAtoms(i, j) is None:  
                mol.AddBond(i, j, Chem.BondType.SINGLE)  # Only add if not present'
         


neighbors = structure.get_neighbors(structure, r=cutoff)

for site, neighbor_list in zip(structure, neighbors):
    for neighbor in neighbor_list:
        i = structure.index(site)
        j = structure.index(neighbor.site)

        # Ensure atoms are from the same unit cell
        if neighbor.image == (0, 0, 0):  
            dist = neighbor.nn_distance
            print(f"Bond between atom {i} and atom {j}: {dist:.3f} Å")
            if mol.GetBondBetweenAtoms(i, j) is None:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

from rdkit.Chem import AllChem

conf = Chem.Conformer(mol.GetNumAtoms())  # Create a conformer

for idx, site in enumerate(structure):
    coord = site.coords  # Cartesian coordinates (x, y, z)
    conf.SetAtomPosition(idx, coord)  # Assign position

mol.AddConformer(conf)  # Add conformer to molecule
'''

def cif_to_mol(cif_file, cutoff):

    if not cif_file.endswith('.cif'):
        raise ValueError(f"The provided file '{cif_file}' is not a CIF file. Please provide a valid CIF file.")
    
    structure = Structure.from_file(cif_file)

    mol = Chem.RWMol()  # Create an editable molecule in RDKit

    for site in structure:
        element = site.specie.symbol 
        atom = Chem.Atom(element) 
        mol.AddAtom(atom) 

    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            dist = structure.get_distance(i, j)
            if dist < cutoff:  
                if mol.GetBondBetweenAtoms(i, j) is None:  
                    mol.AddBond(i, j, Chem.BondType.SINGLE) 

    conf = Chem.Conformer(mol.GetNumAtoms()) 

    for idx, site in enumerate(structure):
        coord = site.coords 
        conf.SetAtomPosition(idx, coord) 

    mol.AddConformer(conf) 

    return mol

mol = cif_to_mol(cif_file, 1.5)

from rdkit.Chem import Draw

image = Draw.MolToImage(mol)  # 2D visualization
image.show()