import importlib
import torch
from .node import NodeModel, NodeEdgeModel
from ..utils.helpers import one_hot

class LFFlow(torch.nn.Module):
    def __init__(self, n_iter, dequant_network, dt, h_dim, node_hidden_layers, energy_model, box, **params):
        super().__init__()
        self.energy_networks = []
        self.node_networks = []
        self.node_force_networks = []
        
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
        self.to(torch.double)
    
    def forward(self, data):
        ldj = 0
        data.h, ldj = self.dequantize(data.h)
        for Energy_nn, Node_nn, NodeForce_nn in zip(self.energy_networks, self.node_networks, self.node_force_networks):
            data.pos.requires_grad_(True) # to calculate derivate of energy
            
            Q = Node_nn(data.h)
            G = NodeForce_nn(data.h)
            E, F = Energy_nn(data)

            data.vel = torch.exp(Q) * data.vel + F*self.dt
            data.g = data.g + G*self.dt
            print(E)
            data.pos = data.pos + data.vel*self.dt
            data.pbc()
            data.h = data.h + data.g*self.dt

            ldj += Q.sum()

        return data, ldj

    def reverse(self, data):
        for Energy_nn, Node_nn, NodeForce_nn in zip(reversed(self.energy_networks), reversed(self.node_networks), reversed(self.node_force_networks)):
            data.h = data.h - data.g*self.dt
            data.pos = data.pos - data.vel*self.dt
            data.pbc()
        
            Q = Node_nn(data.h)
            G = NodeForce_nn(data.h)
            E, F = Energy_nn(data)
            print(E)
            data.g = data.g - G*self.dt
            data.vel = (data.vel - F*self.dt)/torch.exp(Q)
        
        data.h = self.dequantize.reverse(data.h)
    
        return data