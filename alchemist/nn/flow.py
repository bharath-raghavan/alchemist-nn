import importlib
import torch
from .node import NodeModel, NodeEdgeModel
from torch.autograd import grad
from ..utils.helpers import apply_pbc

class LFFlow(torch.nn.Module):
    def __init__(self, n_iter, dequant_network, dt, h_dim, node_hidden_layers, energy_model, box, prec, **params):
        super().__init__()
        self.energy_networks = []
        self.node_networks = []
        self.node_force_networks = []
        self.box = torch.tensor(box) 
        
        for i in range(n_iter):
            energy_model_class = getattr(importlib.import_module(f"alchemist.nn.energy.{energy_model}"), f"{energy_model.upper()}")
            self.energy_networks.append(energy_model_class(h_dim, box, **params))
            self.node_networks.append(NodeModel(h_dim, 1, node_hidden_layers))
            self.node_force_networks.append(NodeModel(h_dim, h_dim, node_hidden_layers))
        
        self.energy_networks = torch.nn.ModuleList(self.energy_networks) 
        self.node_networks = torch.nn.ModuleList(self.node_networks)
        self.node_force_networks = torch.nn.ModuleList(self.node_force_networks)
        
        self.dequantize = dequant_network
        self.dt = dt
        self.dt_2 = 0.5*dt
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
        for Energy_nn, Node_nn, NodeForce_nn in zip(self.energy_networks, self.node_networks, self.node_force_networks):
            
            data.pos = self.pbc(data.pos + data.vel*self.dt)
            data.h = data.h + data.g*self.dt
            
            Q = Node_nn(data.h)
            G = NodeForce_nn(data.h)
            E = Energy_nn(data)
            F = LFFlow.get_force(E, data.pos)
            
            data.vel = torch.exp(Q) * data.vel + F*self.dt
            data.g = data.g + G*self.dt
            
            ldj += Q.sum()

        return data, ldj

    def reverse(self, data):
        for Energy_nn, Node_nn, NodeForce_nn in zip(reversed(self.energy_networks), reversed(self.node_networks), reversed(self.node_force_networks)):
            
            Q = Node_nn(data.h)
            G = NodeForce_nn(data.h)
            E = Energy_nn(data)
            F = LFFlow.get_force(E, data.pos, reverse=True)
            
            data.g = data.g - G*self.dt
            data.vel = (data.vel - F*self.dt)/torch.exp(Q)
            
            data.h = data.h - data.g*self.dt
            data.pos = self.pbc(data.pos - data.vel*self.dt)
            
        data.h = self.dequantize.reverse(data.h)
        
        return data