import torch
from .base import BaseDataset, InMemoryBaseDataset, Data
import pymolr
from ..utils.helpers import get_element
                
class TRRDataset(BaseDataset):

    def __len__(self):
        u = pymolr.System(mpt_file=self.input_params['top_file'], trr_files=self.input_params['traj_file'])
        return u.nframes

    def process(self, idx):
        dist_units = self.input_params['dist_unit']
        time_units = self.input_params['time_unit']
        
        if dist_units == 'ang':
            dist_scale = 10
        elif dist_units == 'nm':
            dist_scale = 1
            
        if time_units == 'pico':
            time_scale = 1
        elif time_units == 'femto':
            time_scale = 1e-3
                    
        u = pymolr.System(mpt_file=self.input_params['top_file'], trr_files=self.input_params['traj_file'])
        sele = u.select('all', frame=idx)
        z = [get_element(e, m) for e,m in zip(sele.element, sele.mass)]

        return (
            z,
            torch.tensor(sele.positions*dist_scale, dtype=self.dtype),
            torch.tensor(sele.velocities*dist_scale/time_scale, dtype=self.dtype),
            'Frame: ' + str(idx)
        )