import torch

class NetworkWrapper(torch.nn.Module):
    def __init__(self, energy_network, node_network, node_force_network):
        super().__init__()        
        self.energy_network = energy_network
        self.node_network = node_network
        self.node_force_network = node_force_network
    
    def forward(self, data):
        Q = self.node_network(data.h)
        E, edge_index, coord_diff = self.energy_network(data)
        G = self.node_force_network(data.h, edge_index, coord_diff)
        
        return Q, G, E