import torch
from torch import nn
from ...utils.helpers import elem_to_one_hot, log_gaussian
import torch.nn.functional as F
from ..node.scalar import ScalarNodeModel

class ArgMax(ScalarNodeModel):
    def __init__(self, atom_types, dtype, hidden_nf, act_fn=nn.SiLU()):
        node_nf = len(atom_types)
        super().__init__(node_nf, node_nf*2, hidden_nf, act_fn)
        self.atom_types = {z:i for i,z in enumerate(atom_types)}
        self.dtype = dtype
        
    def forward(self, z):
        h = elem_to_one_hot(z, self.atom_types, dtype=self.dtype)
    
        net_out = self.network(h)
        log_scale, translate = torch.chunk(net_out, chunks=2, dim=-1)
        u = translate + torch.randn(h.size(), device=h.device) * log_scale.exp()
        #u = torch.randn(h.size(), device=h.device)
        log_q = log_gaussian(u) - log_scale.sum()
        
        T = (h * u).sum(-1, keepdim=True)
        z = h * u + (1 - h) * (T - F.softplus(T - u))
        ldj = (1 - h) * F.logsigmoid(T - u)
        log_q = log_q - ldj.sum()
        
        return z, log_q
    
    def reverse(self, z):
        return torch.argmax(z, dim=-1)