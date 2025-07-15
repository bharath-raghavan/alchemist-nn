from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.autograd import grad
from torch.nn import Embedding
from torch_geometric.nn.models import visnet
from torch_geometric.utils import scatter

class Distance(torch.nn.Module):
    def __init__(
        self, add_self_loops: bool = False
    ) -> None:
        super().__init__()
        self.add_self_loops = add_self_loops

    def forward(
        self,
        edge_index, edge_vec
    ) -> Tuple[Tensor, Tensor, Tensor]:

        if self.add_self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device, dtype=edge_vec.dtype)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec
        
class ViSNetBlock(visnet.ViSNetBlock):
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        vertex: bool = False,
        cutoff: float = 5.0,
    ) -> None:
    
        dummy_max_num_neighbors = 0
        super().__init__(lmax, vecnorm_type, trainable_vecnorm, num_heads, num_layers, hidden_channels, num_rbf, trainable_rbf, max_z, cutoff, dummy_max_num_neighbors, vertex)

        self.distance = Distance()
            
class VISNET(visnet.ViSNet):
    def __init__(
           self,
           max_z: int,
           lmax: int = 1,
           vecnorm_type: Optional[str] = None,
           trainable_vecnorm: bool = False,
           num_heads: int = 8,
           num_layers: int = 6,
           hidden_channels: int = 128,
           num_rbf: int = 32,
           trainable_rbf: bool = False,
           vertex: bool = False,
           atomref: Optional[Tensor] = None,
           reduce_op: str = "sum",
           cutoff: float = 5.0,
           mean: float = 0.0,
           std: float = 1.0
       ) -> None:
       
       super().__init__()
       
       self.representation_model = ViSNetBlock(
           lmax=lmax,
           vecnorm_type=vecnorm_type,
           trainable_vecnorm=trainable_vecnorm,
           num_heads=num_heads,
           num_layers=num_layers,
           hidden_channels=hidden_channels,
           num_rbf=num_rbf,
           trainable_rbf=trainable_rbf,
           max_z=max_z,
           cutoff=cutoff,
           vertex=vertex)
           
       self.output_model = visnet.EquivariantScalar(hidden_channels=hidden_channels)
       self.prior_model = visnet.Atomref(atomref=atomref, max_z=max_z)
       self.reduce_op = reduce_op

       self.register_buffer('mean', torch.tensor(mean))
       self.register_buffer('std', torch.tensor(std))

       self.reset_parameters()
    
    def get_batch(self, data):
       batch = []
       for i, mol in enumerate(data):
           batch.append([i]*mol.num_atoms)
       
       return torch.tensor(batch, device=data.device).flatten()
    
    def forward(self, data) -> Tuple[Tensor, Optional[Tensor]]:
       edge_index = torch.stack((data.edges.row, data.edges.col))
       z = data.h.argmax(dim=1)
       
       x, v = self.representation_model(z,  edge_index, data.edges.coord_diff) # pass edge_index and coord_diff in place of pos and batch
       x = self.output_model.pre_reduce(x, v)
       x = x * self.std

       if self.prior_model is not None:
           x = self.prior_model(x, z)
       
       y = scatter(x, self.get_batch(data), dim=0, reduce=self.reduce_op)
       y = y + self.mean

       grad_outputs = [torch.ones_like(y)]
       dy = grad(
           [y],
           [data.pos],
           grad_outputs=grad_outputs,
           create_graph=True,
           retain_graph=True,
       )[0]
       if dy is None:
           raise RuntimeError(
               "Autograd returned None for the force prediction.")
       return y, -dy
