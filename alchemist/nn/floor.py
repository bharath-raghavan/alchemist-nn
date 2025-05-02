import torch
from ..utils.helpers import one_hot, log_gaussian
import torch.nn.functional as F

class Floor(torch.nn.Module):
    def __init__(self, dequant_scale=1):
        super().__init__()
        self.dequant_scale = dequant_scale
        
    def forward(self, z):
        z = z.to(torch.float64)
        return z + self.dequant_scale*torch.rand_like(z).detach(), 0
    
    def reverse(self, z): return torch.floor(z)