import importlib
import torch
from .node import NodeModel, NodeEdgeModel

class LFFlow(torch.nn.Module):
    def __init__(self, n_iter, dequant_network, dt, h_dim, node_hidden_layers, energy_model, **params):
        super().__init__()
        self.energy_networks = []
        self.node_networks = []
        self.node_edge_networks = []
        
        for i in range(n_iter):
            energy_model_class = getattr(importlib.import_module(f"alchemist.nn.energy.{energy_model}"), f"{energy_model.upper()}")
            self.energy_networks.append(energy_model_class(h_dim, **params))
            self.node_networks.append(NodeModel(h_dim, 1, node_hidden_layers))
            self.node_edge_networks.append(NodeEdgeModel(h_dim, node_hidden_layers))
        
        self.energy_networks = torch.nn.ModuleList(self.energy_networks) 
        self.node_networks = torch.nn.ModuleList(self.node_networks)
        self.node_edge_networks = torch.nn.ModuleList(self.node_edge_networks)
        
        self.dequantize = dequant_network
        self.dt = dt
        self.dt_2 = 0.5*dt
        self.to(torch.double)

    def forward(self, data):
        data.h, ldj = self.dequantize(data.h)
        for Energy_nn, Node_nn, NodeEdge_nn in zip(self.energy_networks, self.node_networks, self.node_edge_networks):
            data.pos.requires_grad_(True) # to calculate derivate of energy
            
            Q = Node_nn(data.h)
            G = NodeEdge_nn(data.h, data.edges)
            E, F = Energy_nn(data)
            data.vel = torch.exp(Q) * data.vel + F*self.dt
            data.g = data.g + G*self.dt
        
            data.pos = data.pos + data.vel*self.dt
            data.pbc()
            data.h = data.h + data.g*self.dt

            ldj += Q.sum()

        return data, ldj

    def reverse(self, data):
        for Energy_nn, Node_nn, NodeEdge_nn in zip(self.energy_networks, self.node_networks, self.node_edge_networks):
            data.pos.requires_grad_(True) # to calculate derivate of energy
            
            data.h = data.h - data.g*self.dt
            data.pos = data.pos - data.vel*self.dt
            data.pbc()
        
            Q = Node_nn(data.h)
            G = NodeEdge_nn(data.h, data.edges)
            E, F = Energy_nn(data)
            data.g = data.g - G*self.dt
            data.vel = (data.vel - F*self.dt)/torch.exp(Q)
        
        data.h = self.dequantize.reverse(data.h)
    
        return data