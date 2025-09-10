import importlib
import torch

from ...utils.helpers import apply_pbc
from ..embedding.default import Default
from ase.neighborlist import primitive_neighbor_list

class FlowModel(torch.nn.Module):
    def __init__(self, energy_networks, node_nf, dim, dt, cutoff, box, dtype=torch.double):
        super().__init__()
        self.cell = [[box[0], 0, 0],[0, box[1], 0], [0, 0, box[2]]]
        self.register_buffer('box', torch.tensor(box))
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('dt_2', torch.tensor(0.5*dt))
        
        self.energy_networks = torch.nn.ModuleList(energy_networks)
        self.embedding = Default(node_nf, dtype, dim*2)
        
        self.cutoff = cutoff
        
        self.dtype = dtype
        
        self.to(self.dtype)
    
    def pbc(self, coord):
        return torch.cat( [apply_pbc(coord[:, :3], self.box), coord[:, 3:]], dim=1)
    
    def get_edges(self, data):
        edge_index = torch.empty((2, 0), dtype=torch.int64, device=data.device)
        N_cnt = 0
        for mol in data:
            i, j = primitive_neighbor_list("ij", [True, True, True], self.cell, mol.pos.detach().cpu().numpy(), self.cutoff, self_interaction=True)
            edge_index_mol = torch.tensor([(i+N_cnt).tolist(), (j+N_cnt).tolist()], dtype=torch.int64, device=data.device)
            edge_index = torch.cat((edge_index, edge_index_mol), dim=1)
            N_cnt += mol.num_atoms
        
        return edge_index

    
    @staticmethod
    def compute_forces(energy, positions, training = True) -> torch.Tensor:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
        gradient = torch.autograd.grad(
            outputs=[energy], 
            inputs=[positions],
            grad_outputs=grad_outputs,
            retain_graph=training,  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]
        if gradient is None:
            raise RuntimeError("Autograd returned None for the force prediction.")
        return -1 * gradient
    
    def forward(self, data):
        data.pos = self.pbc(data.pos) # put particles in box, just in case it is not TODO: print warning if pos changes here
        data.pos.requires_grad_(True) # to calculate derivate of energy
            
        x, ldj = self.embedding(data.z)
        h, g = x.chunk(2, dim=1)
        
        r = torch.cat([data.pos, data.h], dim=1)
        v = torch.cat([data.vel, data.g], dim=1)
        
        for nn in self.energy_networks:
            r = self.pbc(r + v*self.dt)
            
            data.pos = r[:, :3]
            h = r[:, 3:]
            
            edges = self.get_edges(data)
            E = nn(data, h, edges)
            F = FlowModel.compute_forces(E, r)
            
            v = v + F*self.dt
        
        data.vel = v[:, :3]
        g = v[:, 3:]
        return data, h, g

    def reverse(self, data, x=None):
        if x is None:
            x = torch.normal(0, 1, size=(data.pos.shape[0], self.embedding.dim), dtype=self.dtype, device=data.device)
        
        h, g = x.chunk(2, dim=1)
        
        for nn in reversed(self.networks):
            
            Q, G, E = nn(data, h, g)
            F = FlowModel.get_force(E, data.pos, reverse=True)
            
            g = g - G*self.dt
            data.vel = (data.vel - F*self.dt)/torch.exp(Q)
            
            h = h - g*self.dt
            data.pos = self.pbc(data.pos - data.vel*self.dt)
        
        x = torch.cat([h,g], dim=1)
        data.z = self.embedding.reverse(x)
        
        return data