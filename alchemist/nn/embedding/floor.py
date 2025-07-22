import torch
from ..utils.helpers import one_hot, log_gaussian
import torch.nn.functional as F

class Floor(torch.nn.Module):
    def __init__(self, atom_types, dtype, dequant_scale=1):
        super().__init__()
        self.atom_types = atom_types
        self.dequant_scale = dequant_scale
        self.dtype = dtype
        
    def forward(self, z):
        h = elem_to_one_hot(z, self.atom_types, dtype=self.dtype)
        return z + self.dequant_scale*torch.rand_like(z).detach(), 0
    
    def reverse(self, z): return torch.floor(z)