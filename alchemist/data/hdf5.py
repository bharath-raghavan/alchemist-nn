import torch
from .base import BaseDataset, Data
from ..utils.constants import atom_types

import h5py



class HDF5Dataset(BaseDataset):
    def process(self, **input_params):
        f = h5py.File(input_params['raw_file'], 'r')
        for i in f.keys():
            for j in f[i].keys():
                dct = f[i][j]
                z = [i.decode('utf-8') for i in dct['species']]
                self.append(
                    z=z,
                    pos=torch.tensor(dct['coordinates'][:][0]*self.dist_scale, dtype=torch.float64),
                    box=torch.tensor([dct['cell'][0,0,0]*self.dist_scale,dct['cell'][0,1,1]*self.dist_scale,dct['cell'][0,2,2]*self.dist_scale], dtype=torch.float64),
                    label='hdf5'
                    )
