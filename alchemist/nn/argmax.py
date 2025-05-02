import torch
from torch import nn
from ..utils.helpers import one_hot, log_gaussian
import torch.nn.functional as F

class ArgMax(torch.nn.Module):
    def __init__(self, node_nf, hidden_nf, act_fn=nn.SiLU()):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(node_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, node_nf*2))
        
    def forward(self, h):
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
        return one_hot(torch.argmax(z, dim=-1), dtype=torch.float64)