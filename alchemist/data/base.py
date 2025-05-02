import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from .transforms import NoneTransform
from ..utils.helpers import apply_pbc, get_box_len, get_periodic_images_within, one_hot
from ..utils.constants import atom_types

class Edges: # edge class to handle coord_diff over periodic box
    def __init__(self, edge_index, box, coord):
        self.box = box
        self.row, self.col = edge_index
        self.coord = coord
    
    @property
    def coord_diff(self):
        coord_diff = self.coord[self.row] - self.coord[self.col]
        coord_diff = apply_pbc(coord_diff, self.box*0.5) # get nearest periodic image dist , old method -> coord_diff = coord_diff - (coord_diff>(self.box*0.5))*self.box*0.5
        return coord_diff

class Data:
    def __init__(self, z=None, h=None, g=None, pos=None, vel=None, N=None, r_cut=None, box=None, label=None, device='cpu'):
        self.z = z
        self.h = h
        self.g = g
        self.pos = pos
        self.vel = vel
        self.N = N
        self.r_cut = r_cut
        self.box = box
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
                r_cut=self.r_cut[i],
                box=self.box[start_id:end_id,:],
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
        h = self.h.clone()
        g = self.g.clone()
        pos = self.pos.clone()
        vel = self.vel.clone()
        N = self.N.clone()
        r_cut = self.r_cut.clone()
        box = self.box.clone()
        
        return Data(
                z=self.z,
                h=h,
                g=g,
                pos=pos,
                vel=vel,
                N=N,
                r_cut=r_cut,
                box=box,
                label=self.label,
                device=self.device
            )
            
    def to(self, device):
        h = self.h.to(device)
        g = self.g.to(device)
        pos = self.pos.to(device)
        vel = self.vel.to(device)
        N = self.N.to(device)
        r_cut = self.r_cut.to(device)
        box = self.box.to(device)
        
        return Data(
                z=self.z,
                h=h,
                g=g,
                pos=pos,
                vel=vel,
                N=N,
                r_cut=r_cut,
                box=box,
                label=self.label,
                device=device
            )
    
    def pbc(self):
        self.pos = apply_pbc(self.pos, self.box)
            
    @property
    def edges(self):
        # get neighbour list

        edge_index = torch.empty((2, 0), dtype=torch.int, device=self.device)
        boxes = []
        N_cnt = 0
        for mol in self:
            box = mol.box[0] # assume that each atom in mol has same box lens (should be true), so use only first one
            pos_all_periodic_images, id_mapping = get_periodic_images_within(mol.pos, box, mol.r_cut) # replicate positions 27 times and find those within r_cut
            
            r_sq = mol.r_cut*mol.r_cut
            
            dist_sq = (pos_all_periodic_images.unsqueeze(1) - mol.pos).pow(2).sum(dim=2) # calculating diff with all (27) images takes time
            ids = (dist_sq < r_sq).nonzero()
            periodic_ids = id_mapping[ids]
            edge_index_mol = periodic_ids + N_cnt
            edge_index_mol = edge_index_mol[torch.nonzero(edge_index_mol[:, 0] - edge_index_mol[:, 1])].squeeze(1)
            boxes.append(box.repeat(edge_index_mol.shape[0], 1)) # repeate box size for each edge
            edge_index = torch.cat((edge_index, edge_index_mol.T), dim=1)
            N_cnt += mol.num_atoms
            
        return Edges(edge_index, torch.cat(boxes), self.pos)
        
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
                z=[d.z for d in dataset],
                h=torch.cat([d.h for d in dataset]),
                g=torch.cat([d.g for d in dataset]),
                pos=torch.cat([d.pos for d in dataset]),
                vel=torch.cat([d.vel for d in dataset]),
                N=torch.tensor([d.N for d in dataset]),
                r_cut=torch.tensor([d.r_cut for d in dataset]),
                box=torch.cat([d.box for d in dataset]),
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
        
        if 'box' in input_params:
            self.box = torch.tensor(input_params['box'])
        else:
            self.box = None

        if 'r_cut' in input_params:
            self.r_cut = float(input_params['r_cut'])
            input_params.pop('r_cut')
        else:
            self.r_cut = None
            
        self.input_params = input_params
    
    @abstractmethod   
    def __len__(self):
        pass

    def __getitem__(self, idx):
        z, pos, vel, label = self.process(idx)
        return self._get_data(z, pos, vel, label)
        
    
    def _get_data(self, z, pos, vel, label):
        if self.box is None:
            self.box = get_box_len(pos)
        
        if self.r_cut is None:
            print("error rcut")

        atom_types = {z:i for i,z in enumerate(self.atom_types)}
        type_idx = [atom_types[i] for i in z]
        h = one_hot(torch.tensor(type_idx), num_classes=len(atom_types), dtype=torch.float64)

        if vel is None:
            vel = torch.zeros_like(pos)

        N = pos.shape[0]

        data = Data(
            z=z,
            h=h,
            g=torch.normal(0, 1, size=h.shape, dtype=torch.float64),
            pos=pos,
            vel=vel,
            N=N,
            r_cut=self.r_cut,
            box=self.box.repeat(N, 1), # tile box len to be same size as pos
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
