import importlib
import torch
from torch.autograd import grad
from ...utils.helpers import apply_pbc

class FlowModel(torch.nn.Module):
    def __init__(self, networks, dequant_network, dt, box, prec):
        super().__init__()
        self.register_buffer('box', torch.tensor(box))
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('dt_2', torch.tensor(0.5*dt))
        
        self.networks = torch.nn.ModuleList(networks)
        
        self.dequantize = dequant_network

        if prec == 64:
            self.to(torch.double)
        else:
            self.to(torch.float)
    
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
        data.pos.requires_grad_(True) # to calculate derivate of energy
        
        ldj = 0
        data.pos = self.pbc(data.pos) # put particles in box, just in case it is not TODO: print warning if pos changes here
        data.h, ldj = self.dequantize(data.h)
        for nn in self.networks:
            
            data.pos = self.pbc(data.pos + data.vel*self.dt)
            data.h = data.h + data.g*self.dt
            
            Q, G, E = nn(data)
            F = FlowModel.get_force(E, data.pos)
            
            data.vel = torch.exp(Q) * data.vel + F*self.dt
            data.g = data.g + G*self.dt
            
            ldj += Q.sum()

        return data, ldj

    def reverse(self, data):
        for nn in reversed(self.networks):
            
            Q, G, E = nn(data)
            F = FlowModel.get_force(E, data.pos, reverse=True)
            
            data.g = data.g - G*self.dt
            data.vel = (data.vel - F*self.dt)/torch.exp(Q)
            
            data.h = data.h - data.g*self.dt
            data.pos = self.pbc(data.pos - data.vel*self.dt)
            
        data.h = self.dequantize.reverse(data.h)
        
        return data