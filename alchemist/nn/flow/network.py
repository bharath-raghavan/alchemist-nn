import torch

class NetworkWrapper(torch.nn.Module):
    def __init__(self, energy_network, node_network, node_force_network):
        super().__init__()        
        self.energy_network = energy_network
        self.node_network = node_network
        self.node_force_network = node_force_network
    
    def forward(self, data, h):
        Q = self.node_network(h)
        E, edges = self.energy_network(data, h)
        G = self.node_force_network(h, edges)
        
        return Q, G, E