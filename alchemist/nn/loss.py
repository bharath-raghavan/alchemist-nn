import math
import torch
from ..utils.helpers import log_gaussian

class Alchemical_NLL:
    def __init__(self, kBT, partition_func=10, softening=0):
        self.kBT = kBT
        self.z_lj = partition_func
        self.softening = softening
        
    def _get_lj_potential(self, data):
        H = 0
        for mol in data:
            dist_sq = torch.triu((mol.pos.unsqueeze(1) - mol.pos).pow(2).sum(dim=2))
            r_sq = dist_sq[dist_sq != 0] + self.softening
            r_6 = r_sq.pow(3)
            r_12 = r_6.pow(2)
            H += 4*(1/r_12 - 1/r_6).sum()
        return H 

    def __call__(self, out, ldj):
        H = self._get_lj_potential(out) + 0.5*(out.vel**2).sum()
        logZ = - out.num_atoms*( math.log(self.z_lj) - 1.5*math.log(2*math.pi/self.kBT))
        log_px = - H/self.kBT + logZ + ldj + log_gaussian(out.h) + log_gaussian(out.g)
        return -log_px/out.num_mols