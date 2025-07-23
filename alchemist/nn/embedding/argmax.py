import torch
from torch import nn
from ...utils.helpers import one_hot, log_gaussian
import torch.nn.functional as F
from ..node.scalar import ScalarNodeModel

class Argmax(ScalarNodeModel):
    def __init__(self, node_nf, dtype, hidden_nf, dim, act_fn=nn.SiLU()):
        if node_nf > dim:
            print("error")
        super().__init__(dim, dim*2, hidden_nf, act_fn)
        self.dtype = dtype
        self.dim = dim
    
    def forward(self, z):
        atom_feat = one_hot(z, num_classes=self.dim, dtype=self.dtype)
        
        net_out = self.network(atom_feat)
        log_scale, translate = torch.chunk(net_out, chunks=2, dim=-1)
        u = translate + torch.randn(h.size(), device=h.device) * log_scale.exp()
        
        log_q = log_gaussian(u) - log_scale.sum()
        
        T = (h * u).sum(-1, keepdim=True)
        out = h * u + (1 - h) * (T - F.softplus(T - u))
        ldj = (1 - h) * F.logsigmoid(T - u)
        log_q = log_q - ldj.sum()
        
        return out, log_q
    
    def reverse(self, z):
        return torch.argmax(z, dim=-1)