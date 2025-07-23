import torch
from ..utils.helpers import one_hot, log_gaussian
import torch.nn.functional as F

class Floor(torch.nn.Module):
    def __init__(self, node_nf, dtype, dequant_scale=1):
        super().__init__()
        if node_nf > dim:
            print("error")
        self.dim = dim
        self.dequant_scale = dequant_scale
        self.dtype = dtype
                    
    def forward(self, z):
        x = one_hot(z, num_classes=self.dim, dtype=self.dtype)
        return x + self.dequant_scale*torch.rand_like(x).detach(), 0
    
    def reverse(self, x): return torch.argmax(torch.floor(x), dim=-1)