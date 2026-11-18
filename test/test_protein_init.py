from models.protein_init import protein_init
from models.ligand_init import ligand_init
import pandas as pd
import torch

test = pd.read_csv('data/test.csv')
protein_seqs = list(set(test['Protein'].tolist()))
ligand_smiles = list(set(test['Ligand'].to_list()))

protein_dict = protein_init(protein_seqs)
torch.save(protein_dict, 'protein.pt')

ligand_dict = ligand_init(ligand_smiles)
torch.save(ligand_dict, 'ligand.pt')