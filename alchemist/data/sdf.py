import torch
from .base import BaseDataset, Data
from rdkit import Chem
from tqdm import tqdm
from ..utils.constants import atom_types
from ..utils.helpers import one_hot

class SDFDataset(BaseDataset):
    
    def process(self, **input_params):
        suppl = Chem.SDMolSupplier(input_params['raw_file'], removeHs=False, sanitize=False)

        for mol in tqdm(suppl):
            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float64)
    
            z = [atom.GetSymbol() for atom in mol.GetAtoms()]
            type_idx = [atom_types[i] for i in z]
            h = one_hot(torch.tensor(type_idx), num_classes=len(atom_types), dtype=torch.float64)

            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            
            label = name + ' ' + smiles
    
            self.append(
                z=z,
                h=h,
                pos=pos,
                N=N,
                label=label
            )
