from typing import Optional, List
import torch
from visnet.models.model import ViSNet
from visnet.models.visnet_block import ViSNetBlock
from visnet.models.output_modules import EquivariantScalar
from .dist import PeriodicDistance

class MockEmbedding(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def reset_parameters(self):
        pass

    def forward(self, z): # dummy z
        return self.h
    

class Visnet(ViSNet):
    def __init__(
           self,
           box: List,
           lmax: int = 1,
           vecnorm_type: Optional[str] = None,
           trainable_vecnorm: bool = False,
           num_heads: int = 8,
           num_layers: int = 6,
           hidden_channels: int = 128,
           num_rbf: int = 32,
           rbf_type="expnorm",
           trainable_rbf: bool = False,
           activation="silu",
           attn_activation="silu",
           cutoff: float = 5.0,
           max_num_neighbors: int = 32,
           vertex_type: str = "None",
           atomref: Optional[List] = None,
           reduce_op: str = "sum",
           loops: bool = True
       ) -> None:
       
       representation_model = ViSNetBlock(lmax,
        vecnorm_type,
        trainable_vecnorm,
        num_heads,
        num_layers,
        hidden_channels,
        num_rbf,
        rbf_type,
        trainable_rbf,
        activation,
        attn_activation,
        100, # max_z dummy
        cutoff,
        max_num_neighbors,
        vertex_type,
       )
       representation_model.distance = PeriodicDistance(cutoff, box, max_num_neighbors=max_num_neighbors, add_self_loops=loops) # replace distance calculator with custom periodic dist version
       representation_model.embedding = MockEmbedding()
       representation_model.neighbor_embedding.embedding = MockEmbedding()
       
       output_model = EquivariantScalar(hidden_channels=hidden_channels)
       
       super().__init__(representation_model,
        output_model,
        None,
        reduce_op,
        torch.scalar_tensor(0.0), # mean
        torch.scalar_tensor(1.0)) # std
    
    def forward(self, data, h):
       self.representation_model.embedding.h = h
       self.representation_model.neighbor_embedding.embedding.h = h
       return super().forward(data)[0], self.representation_model.distance.edges
