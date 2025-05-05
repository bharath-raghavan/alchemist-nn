import math
import torch

class BaseFlow(torch.nn.Module):
    def __init__(self, networks, dequant_network, dt):
        super().__init__()
        self.networks = torch.nn.ModuleList(networks)
        self.dequantize = dequant_network
        self.dt = dt
        self.dt_2 = 0.5*dt
        self.to(torch.double)
        
    def forward(self, data):
        pass

    def reverse(self, data):
        pass
