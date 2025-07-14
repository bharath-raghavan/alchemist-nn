import torch
from .base import BaseDataset, InMemoryBaseDataset, Data
import MDAnalysis
from MDAnalysis.lib.log import ProgressBar
from ..utils.helpers import get_element

class LargeMDDataset(BaseDataset):
       
    def __len__(self):
        u = MDAnalysis.Universe(self.input_params['top_file'], self.input_params['traj_file'])
        return len(u.trajectory)
    
    def process(self, idx):
        u = MDAnalysis.Universe(self.input_params['top_file'], self.input_params['traj_file'])
        u.trajectory[idx]
        z = [get_element(a.element, a.mass) for a in u.atoms]

        return (
            z,
            torch.tensor(u.atoms.positions, dtype=torch.float64),
            torch.tensor(u.atoms.velocities, dtype=torch.float64),
            'Frame: ' + str(idx)
        )
                
                
class MDDataset(InMemoryBaseDataset):

    def process(self, **input_params):

        for top, traj in zip(input_params['top_file'], input_params['traj_file']):
            u = MDAnalysis.Universe(top,  traj)
            for frame, ts in enumerate(ProgressBar(u.trajectory)):
                z = [get_element(a.element, a.mass) for a in u.atoms]

                self.append(
                    z=z,
                    pos=torch.tensor(u.atoms.positions*self.dist_scale, dtype=torch.float64),
                    vel=torch.tensor(u.atoms.velocities*self.dist_scale/self.time_scale, dtype=torch.float64),
                    N=len(u.atoms),
                    label=traj + ' frame: ' + str(frame)
                )
