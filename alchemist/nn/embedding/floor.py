import torch
from ..utils.helpers import one_hot, log_gaussian
import torch.nn.functional as F

class Floor(torch.nn.Module):
    def __init__(self, node_nf, dtype, dequant_scale=1):
        super().__init__()
        self.node_nf = node_nf
        self.dequant_scale = dequant_scale
        self.dtype = dtype

    @property
    def out_dim(self):
        return self.node_nf
                    
    def forward(self, z):
        h = one_hot(z, num_classes=self.node_nf, dtype=self.dtype)
        return z + self.dequant_scale*torch.rand_like(z).detach(), 0
    
    def reverse(self, z): return torch.floor(z)