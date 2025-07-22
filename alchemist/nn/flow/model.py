import importlib
import torch
from torch.autograd import grad
from ...utils.helpers import apply_pbc

class FlowModel(torch.nn.Module):
    def __init__(self, networks, embedding, dt, box, dtype):
        super().__init__()
        self.register_buffer('box', torch.tensor(box))
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('dt_2', torch.tensor(0.5*dt))
        
        self.networks = torch.nn.ModuleList(networks)
        
        self.embedding = embedding
        
        self.dtype = dtype
        
        self.to(self.dtype)
    
    @staticmethod
    def get_force(y, pos, reverse=False):
        grad_outputs = [torch.ones_like(y)]
        dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=not reverse,
                retain_graph=not reverse,
            )[0]
        if dy is None:
            raise RuntimeError(
                "Autograd returned None for the force prediction.")
        return -dy
    
    def pbc(self, pos):
        return apply_pbc(pos, self.box)
    
    def forward(self, data):
        data.pos = self.pbc(data.pos) # put particles in box, just in case it is not TODO: print warning if pos changes here
        data.pos.requires_grad_(True) # to calculate derivate of energy
        
        h, ldj = self.embedding(data.z)
        g = torch.normal(0, 1, size=h.shape, dtype=self.dtype)
        
        for nn in self.networks:
            
            data.pos = self.pbc(data.pos + data.vel*self.dt)
            h = h + g*self.dt
            
            Q, G, E = nn(data, h)
            F = FlowModel.get_force(E, data.pos)
            
            data.vel = torch.exp(Q) * data.vel + F*self.dt
            g = g + G*self.dt
            
            ldj += Q.sum()

        return data, torch.cat([h,g], dim=1), ldj

    def reverse(self, data, x=None):
        if x != None:
            h, g = x.chunk(2, dim=1)
        else:
            size = (data.pos.shape[0], self.networks[0].node_network.network[0].weight.shape[1])
            h = torch.normal(0, 1, size=size, dtype=self.dtype)
            g = torch.normal(0, 1, size=h.shape, dtype=self.dtype)
        
        for nn in reversed(self.networks):
            
            Q, G, E = nn(data, h)
            F = FlowModel.get_force(E, data.pos, reverse=True)
            
            g = g - G*self.dt
            data.vel = (data.vel - F*self.dt)/torch.exp(Q)
            
            h = h - g*self.dt
            data.pos = self.pbc(data.pos - data.vel*self.dt)
        
        data.z = self.embedding.reverse(h)
        
        return data