import torch
from .base import BaseFlow

class LeapFrogIntegrator(BaseFlow):
    def make_networks(self, network):
        networks = []
        for i in range(self.n_iter): networks.append(network)
        return networks

    def forward(self, data):
        ldj = 0
        data.h = self.dequantize(data.h)
        
        for network in self.networks:
            edges = data.get_edges(self.r_cut)
            Q, F, G = network(data.h, edges, data.pos)
            
            data = self.data_to_fixed(data)
            
            data.vel = torch.exp(Q) * data.vel + self.to_fixed(F*self.dt)
            data.g = data.g + self.to_fixed(G*self.dt)
            
            data.pos = data.pos + (data.vel*self.dt).type(data.pos.dtype)
            data.h = data.h + (data.g*self.dt).type(data.h.dtype)
            
            data = self.data_from_fixed(data)
            data.pbc(self.box)
            
            ldj += Q.sum()

        return data, ldj

    def reverse(self, data):
        for network in reversed(self.networks):
            data = self.data_to_fixed(data)
        
            data.h = data.h - (data.g*self.dt).type(data.h.dtype)
            data.pos = data.pos - (data.vel*self.dt).type(data.pos.dtype)
            
            data = self.data_from_fixed(data)
            data.pbc(self.box)
            
            edges = data.get_edges(self.r_cut)
            Q, F, G = network(data.h, edges, data.pos)
            
            data = self.data_to_fixed(data)
            data.g = data.g - self.to_fixed(G*self.dt)
            data.vel = (data.vel - self.to_fixed(F*self.dt))/torch.exp(Q)
            data = self.data_from_fixed(data)
            
        data.h = self.quantize(data.h)

        return data
        
class VelocityVerletIntegrator(BaseFlow):
    def make_networks(self, network):
        networks = []
        for i in range(self.n_iter+1): networks.append(network)
        return networks
    
    def forward(self, data):
        ldj = 0
        data.h = self.dequantize(data.h)
        
        edges = data.get_edges(self.r_cut)
        Q, F, G = self.networks[0](data.h, edges, data.pos)
        for i in range(1,self.n_iter+1):
            scale = 0.5*(1+torch.exp(Q))
            data.vel = scale*data.vel + F*self.dt_2
            data.g = data.g + G*self.dt_2
            
            data.pos = data.pos + data.vel*self.dt
            data.pbc(self.box)
            data.h = data.h + data.g*self.dt
            
            edges = data.get_edges(self.r_cut)
            Q, F, G = self.networks[i](data.h, edges, data.pos)
            scale = 0.5*(torch.exp(Q)-1)
            data.vel = (data.vel + F*self.dt_2)/(1 - scale)
            data.g = data.g + G*self.dt_2
            
            ldj += Q
        return data, ldj.sum()

    def reverse(self, data):
        edges = data.get_edges(self.r_cut)
        Q, F, G = self.networks[self.n_iter](data.h, edges, data.pos)
        for i in reversed(range(0,self.n_iter)):
            data.g = data.g - G*self.dt_2
            scale = 0.5*(torch.exp(Q)-1)
            data.vel = data.vel*(1 - scale) - F*self.dt_2
            
            data.h = data.h - data.g*self.dt
            data.pos = data.pos - data.vel*self.dt
            data.pbc(self.box)
            
            edges = data.get_edges(self.r_cut)
            Q, F, G = self.networks[i](data.h, edges, data.pos)
            
            data.g = data.g - G*self.dt_2
            scale = 0.5*(1+torch.exp(Q))
            data.vel = (data.vel - F*self.dt_2)/scale
            
        data.h = self.quantize(data.h)
        return data
