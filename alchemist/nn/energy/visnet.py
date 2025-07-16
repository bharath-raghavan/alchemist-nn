from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.autograd import grad
from torch.nn import Embedding
from torch_geometric.nn.models import visnet
from torch_geometric.utils import scatter
from torch_geometric.nn import radius_graph
import itertools
from ...utils.helpers import get_periodic_images_within

class Distance(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        box: Tensor,
        max_num_neighbors: int = 32,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.box = box
        self.max_num_neighbors = max_num_neighbors
        self.add_self_loops = add_self_loops
        
    def forward(
        self,
        pos: Tensor,
        batch: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Computes the pairwise distances between atoms in the molecule.

        Args:
            pos (torch.Tensor): The positions of the atoms in the molecule.
            batch (torch.Tensor): A batch vector, which assigns each node to a
                specific example.

        Returns:
            edge_index (torch.Tensor): The indices of the edges in the graph.
            edge_weight (torch.Tensor): The distances between connected nodes.
            edge_vec (torch.Tensor): The vector differences between connected
                nodes.
        """
        
        pos_all_periodic_images, id_mapping, periodic_batch = get_periodic_images_within(pos, self.box, self.cutoff, batch) # replicate positions 27 times and find those within cutoff
        
        edge_index = radius_graph(
            pos_all_periodic_images,
            r=self.cutoff,
            batch=periodic_batch,
            loop=self.add_self_loops,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_index[0] = id_mapping[edge_index[0]]
        edge_index[1] = id_mapping[edge_index[1]]
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.add_self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device, dtype=edge_vec.dtype)
            edge_weight[mask] = torch.linalg.vector_norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.linalg.vector_norm(edge_vec.clone(), dim=-1)

        return edge_index, edge_weight, edge_vec

class ViSNetBlock(visnet.ViSNetBlock):
    def __init__(
        self,
        box: Tensor,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        add_self_loops: bool = True
    ) -> None:

        super().__init__(lmax, vecnorm_type, trainable_vecnorm, num_heads, num_layers, hidden_channels, num_rbf, trainable_rbf, max_z, cutoff, max_num_neighbors, vertex)

        self.distance = Distance(cutoff, box, max_num_neighbors=max_num_neighbors, add_self_loops=add_self_loops)
        self.reset_parameters()
    

class VISNET(visnet.ViSNet):
    def __init__(
           self,
           max_z: int,
           box: Tensor,
           lmax: int = 1,
           vecnorm_type: Optional[str] = None,
           trainable_vecnorm: bool = False,
           num_heads: int = 8,
           num_layers: int = 6,
           hidden_channels: int = 128,
           num_rbf: int = 32,
           trainable_rbf: bool = False,
           cutoff: float = 5.0,
           max_num_neighbors: int = 32,
           vertex: bool = False,
           atomref: Optional[Tensor] = None,
           reduce_op: str = "sum",
           mean: float = 0.0,
           std: float = 1.0,
           add_self_loops: bool = True
       ) -> None:
       
       super().__init__(lmax,
        vecnorm_type,
        trainable_vecnorm,
        num_heads,
        num_layers,
        hidden_channels,
        num_rbf,
        trainable_rbf,
        max_z,
        cutoff,
        max_num_neighbors,
        vertex,
        atomref,
        reduce_op,
        mean,
        std,
        True)
       
       self.representation_model = ViSNetBlock(box,
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
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
       )
    
    def get_batch(self, data):
       batch = []
       for i, mol in enumerate(data):
           batch.append([i]*mol.num_atoms)
       
       return torch.tensor(batch, device=data.device).flatten()
    
    def forward(self, data):
       z = data.h.argmax(dim=1)
       return super().forward(z, data.pos, self.get_batch(data)) 
