from typing import Optional, List
import torch
from torch_geometric.nn.models import visnet
from torch import Tensor

class MockEmbedding(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def reset_parameters(self):
        pass

    def forward(self, z): # dummy z
        return self.h
    
class MockDistance(torch.nn.Module):
    def __init__(self,) -> None:
        super().__init__()

    def forward(
        self,
        pos: Tensor,
        batch: Tensor
    ):
        return self.edge_index, self.edge_weight, self.edge_vec

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
        cutoff: float = 5.0,
        vertex: bool = False
    ) -> None:

        super().__init__(lmax, vecnorm_type, trainable_vecnorm, num_heads, num_layers, hidden_channels, num_rbf, trainable_rbf, 100, cutoff, 32, vertex)

        self.distance = MockDistance()
        #self.embedding = MockEmbedding()
        #self.neighbor_embedding.embedding = MockEmbedding()
        self.reset_parameters()
        
class Visnet(visnet.ViSNet):
    def __init__(
           self,
           node_nf=6,
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
           vertex: bool = False,
           cutoff: float = 5.0,
           vertex_type: str = "None",
           atomref: Optional[List] = None,
           reduce_op: str = "sum"
       ) -> None:
       
       super().__init__(lmax,
               vecnorm_type,
               trainable_vecnorm,
               num_heads,
               num_layers,
               hidden_channels,
               num_rbf,
               trainable_rbf,
               100, # max_z dummy
               cutoff,
               32, # max_num_neighbors
               vertex,
               atomref,
               reduce_op,
               torch.scalar_tensor(0.0), # mean
               torch.scalar_tensor(1.0), # std
               False)
       
       self.representation_model = ViSNetBlock(
                   lmax=lmax,
                   vecnorm_type=vecnorm_type,
                   trainable_vecnorm=trainable_vecnorm,
                   num_heads=num_heads,
                   num_layers=num_layers,
                   hidden_channels=hidden_channels,
                   num_rbf=num_rbf,
                   trainable_rbf=trainable_rbf,
                   vertex=vertex,
                   cutoff=cutoff,
              )
       if atomref is None:
          self.prior_model = None # set prior model to None, otherwise DDP complains about the weights in Atomref not contributing to grad
       else:
          self.prior_model = visnet.Atomref(atomref=atomref, max_z=max_z)
                  
       self.add_self_loops = True
       self.embedding_in = torch.nn.Linear(node_nf+1, hidden_channels)
    
    def forward(self, data, h, edge_index, t):
       time_tensor = torch.tensor([[t]], device=data.device).tile((h.shape[0],)).T
       h = self.embedding_in(torch.cat([h, time_tensor], dim=1))
    
       #self.representation_model.embedding.h = h
       #self.representation_model.neighbor_embedding.embedding.h = h
       
       self.representation_model.distance.edge_index = edge_index
       
       edge_vec = data.pos[edge_index[0]] - data.pos[edge_index[1]]
       self.representation_model.distance.edge_vec = edge_vec
       
       if self.add_self_loops:
           mask = edge_index[0] != edge_index[1]
           edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device, dtype=edge_vec.dtype)
           edge_weight[mask] = torch.linalg.vector_norm(edge_vec[mask], dim=-1)
       else:
           edge_weight = torch.linalg.vector_norm(edge_vec.clone(), dim=-1)
       
       self.representation_model.distance.edge_weight = edge_weight
       
       return super().forward(data.z, data.pos, data.batch)[0]
