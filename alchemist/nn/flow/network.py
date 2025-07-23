import torch

torch.autograd.set_detect_anomaly(True)

class NetworkWrapper(torch.nn.Module):
    def __init__(self, energy_network, node_network, node_force_network):
        super().__init__()        
        self.energy_network = energy_network
        self.node_network = node_network
        self.node_force_network = node_force_network
    
    def forward(self, data, h, g):
        x = torch.cat([h,g], dim=1)
        Q = self.node_network(x)
        E, edges = self.energy_network(data, x)
        G = self.node_force_network(h, edges)
        
        return Q, G, E