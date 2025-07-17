import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from .transforms import NoneTransform
from ..utils.helpers import one_hot
from ..utils.constants import atom_types

class Data:
    def __init__(self, z=None, h=None, g=None, pos=None, vel=None, N=None, label=None, device='cpu'):
        self.z = z
        self.h = h
        self.g = g
        self.pos = pos
        self.vel = vel
        self.N = N
        self.label = label
        self.device = device
        
    def get_mol(self, i):
        if self.N.ndim == 0:
            return self
        start_id = self.N[:i].sum()
        end_id = self.N[i].item()+start_id
        return Data(
                z=self.z[start_id:end_id],
                h=self.h[start_id:end_id,:],
                g=self.g[start_id:end_id,:],
                pos=self.pos[start_id:end_id,:],
                vel=self.vel[start_id:end_id,:],
                N=self.N[i],
                label=self.label[start_id:end_id],
                device=self.device
            )
    
    @property      
    def num_atoms(self):
        return self.N.sum().item()
        
    @property      
    def num_mols(self):
        if self.N.ndim == 0:
            return 1
        else:
            return len(self.N)
        
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.num_mols:
            raise StopIteration
        else:
            mol = self.get_mol(self.i)
            self.i += 1
            return mol
    
    def clone(self):
        z = self.z.clone()
        h = self.h.clone()
        g = self.g.clone()
        pos = self.pos.clone()
        vel = self.vel.clone()
        N = self.N.clone()
        
        return Data(
                z=z,
                h=h,
                g=g,
                pos=pos,
                vel=vel,
                N=N,
                label=self.label,
                device=self.device
            )
            
    def to(self, device):
        z = self.z.to(device)
        h = self.h.to(device)
        g = self.g.to(device)
        pos = self.pos.to(device)
        vel = self.vel.to(device)
        N = self.N.to(device)
        
        return Data(
                z=z,
                h=h,
                g=g,
                pos=pos,
                vel=vel,
                N=N,
                label=self.label,
                device=device
            )
            
    @property
    def batch(self):
       batch_ = []
       for i, mol in enumerate(self):
           batch_.append([i]*mol.num_atoms)
       
       return torch.tensor(batch_, device=self.device).flatten()
        
class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collater,
            **kwargs,
        )
        
    def collater(self, dataset):
        
        return Data(
                z=torch.cat([d.z for d in dataset]),
                h=torch.cat([d.h for d in dataset]),
                g=torch.cat([d.g for d in dataset]),
                pos=torch.cat([d.pos for d in dataset]),
                vel=torch.cat([d.vel for d in dataset]),
                N=torch.tensor([d.N for d in dataset]),
                label=[d.label for d in dataset]
            )

class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, **input_params):
        if 'transform' in input_params:
            self.transform = input_params['transform']
            input_params.pop('transform')
        else:
            self.transform = NoneTransform()
        
        if 'atom_types' in input_params:
            self.atom_types = input_params['atom_types']
        else:
            self.atom_types = None
            
        self.dtype = torch.double
         
        if 'prec' in input_params:
            if int(input_params['prec']) == 32:
                self.dtype = torch.float
            
        self.input_params = input_params
    
    @abstractmethod   
    def __len__(self):
        pass

    def __getitem__(self, idx):
        z, pos, vel, label = self.process(idx)
        return self._get_data(z, pos, vel, label)
        
    
    def _get_data(self, z, pos, vel, label):
        atom_types = {z:i for i,z in enumerate(self.atom_types)}
        type_idx = [atom_types[i] for i in z]
        h = one_hot(torch.tensor(type_idx), num_classes=len(atom_types), dtype=self.dtype)

        if vel is None:
            vel = torch.zeros_like(pos)

        N = pos.shape[0]

        data = Data(
            z=torch.tensor(type_idx),
            h=h,
            g=torch.normal(0, 1, size=h.shape, dtype=self.dtype),
            pos=pos,
            vel=vel,
            N=N,
            label=label
        )
    
        return self.transform(data)
    
    @property  
    def node_nf(self):
        return len(self.atom_types)

    @abstractmethod
    def process(self, **input_params):
        pass


class InMemoryBaseDataset(BaseDataset, ABC):
    def __init__(self, **input_params):
        super().__init__(**input_params)
            
        self.data_list = []
        
        if 'processed_file' in input_params:
            processed_file = input_params['processed_file']
            input_params.pop('processed_file')
            
            if os.path.exists(processed_file):
                self.data_list = torch.load(processed_file, weights_only=False)
            else:
                self.process(**input_params)
                torch.save(self.data_list, processed_file)
        else:
            self.process(**input_params)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    @property  
    def node_nf(self):
        return self.data_list[0].h.shape[1]
    
    @property  
    def num_atoms_per_mol(self):
        return self.data_list[0].N
    
    def append(self, z, pos, vel=None, label=None):
        self.data_list.append(self._get_data(z, pos, vel, label))

class ComposeInMemoryDatasets(InMemoryBaseDataset):
    def __init__(self, datasets):
        self.data_list = []
        
        for i in datasets:
            if self.data_list != []:
                if self.node_nf != i.node_nf:
                    print("error")
            self.data_list += i.data_list
        
    def process(self, **input_params):
        raise NotImplementedError
