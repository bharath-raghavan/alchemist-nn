from typing import Tuple, List
import torch
from torch import Tensor
from torch_geometric.nn import radius_graph

def get_periodic_images_within(pos, box, r_cut, batch):
    # find all 27 periodic images of positions
    pos_all_periodic_images = torch.cat([pos+torch.tensor([a,b,c], dtype=pos.dtype, device=pos.device) for c in [-box[2],box[2],0] for b in [-box[1],box[1],0] for a in [-box[0],box[0],0]])
    batch = batch.repeat(27)
    
    # find positions within box and r_cut
    ellipse_radii = box+r_cut
    scaled_points = pos_all_periodic_images/ellipse_radii
    ids_in_ellipse = torch.sum(scaled_points**2, axis=1) <= 1
    pos_all_periodic_images = pos_all_periodic_images[ids_in_ellipse]
    batch = batch[ids_in_ellipse]
    
    # and get corresponding id mapping from pos to pos_all_periodic_images
    id_mapping = torch.tensor(list(range(0, pos.shape[0])), device=pos.device).repeat(27)
    id_mapping = id_mapping[ids_in_ellipse]
    
    return pos_all_periodic_images, id_mapping, batch

class Edges:
    def __init__(self, index, coord_diff):
        self.row = index[0]
        self.col = index[1]
        self.coord_diff = coord_diff

class PeriodicDistance(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        box: List,
        max_num_neighbors: int = 32,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer('box', torch.tensor(box))
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
        
        self.edges = Edges(edge_index, edge_vec)
        
        return edge_index, edge_weight, edge_vec