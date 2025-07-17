from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.autograd import grad
from torch.nn import Embedding
from torch_geometric.nn.models import visnet
from torch_geometric.utils import scatter
from torch_geometric.nn import radius_graph
import itertools
from .dist import PeriodicDistance

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

        self.distance = PeriodicDistance(cutoff, box, max_num_neighbors=max_num_neighbors, add_self_loops=add_self_loops)
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
           atomref: Optional[List] = None,
           reduce_op: str = "sum",
           mean: float = 0.0,
           std: float = 1.0,
           add_self_loops: bool = True
       ) -> None:
       
       atomref = torch.tensor(atomref)
       
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
        False)
       
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
    
    def forward(self, data):
       z = data.h.argmax(dim=1)
       return super().forward(z, data.pos, data.batch)[0]
