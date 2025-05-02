import torch
from .base import BaseFlow

class LFIntegrator(BaseFlow):
    def make_networks(self, network):
        networks = []
        for i in range(self.n_iter): networks.append(network)
        return networks

    def forward(self, data):
        data.h, ldj = self.dequantize(data.h)
        for network in self.networks:
            Q, F, G = network(data.h, data.edges)
            data.vel = torch.exp(Q) * data.vel + F*self.dt
            data.g = data.g + G*self.dt
            
            data.pos = data.pos + data.vel*self.dt
            data.pbc()
            data.h = data.h + data.g*self.dt

            ldj += Q.sum()

        return data, ldj

    def reverse(self, data):
        for network in reversed(self.networks):
            data.h = data.h - data.g*self.dt
            data.pos = data.pos - data.vel*self.dt
            data.pbc()
            
            Q, F, G = network(data.h, data.edges)
            data.g = data.g - G*self.dt
            data.vel = (data.vel - F*self.dt)/torch.exp(Q)
            
        data.h = self.dequantize.reverse(data.h)
        
        return data
        
class VVIntegrator(BaseFlow):
    def make_networks(self, network):
        networks = []
        for i in range(self.n_iter+1): networks.append(network)
        return networks
    
    def forward(self, data):
        ldj = 0
        data.h = self.dequantize(data.h)
        
        Q, F, G = self.networks[0](data.h, data.edges)
        ldj += Q
        for i in range(1,self.n_iter+1):
            scale = 0.5*(1+torch.exp(Q))
            data.vel = scale*data.vel + F*self.dt_2
            data.g = data.g + G*self.dt_2
            
            data.pos = data.pos + data.vel*self.dt
            data.pbc()
            data.h = data.h + data.g*self.dt
            
            Q, F, G = self.networks[i](data.h, data.edges)
            scale = 0.5*(torch.exp(Q)-1)
            data.vel = (data.vel + F*self.dt_2)/(1 - scale)
            data.g = data.g + G*self.dt_2
            
            ldj += Q
        return data, ldj.sum()

    def reverse(self, data):
        Q, F, G = self.networks[self.n_iter](data.h, data.edges)
        for i in reversed(range(0,self.n_iter)):
            data.g = data.g - G*self.dt_2
            scale = 0.5*(torch.exp(Q)-1)
            data.vel = data.vel*(1 - scale) - F*self.dt_2
            
            data.h = data.h - data.g*self.dt
            data.pos = data.pos - data.vel*self.dt
            data.pbc()

            Q, F, G = self.networks[i](data.h, data.edges)
            
            data.g = data.g - G*self.dt_2
            scale = 0.5*(1+torch.exp(Q))
            data.vel = (data.vel - F*self.dt_2)/scale
            
        data.h = self.quantize(data.h)
        return data
